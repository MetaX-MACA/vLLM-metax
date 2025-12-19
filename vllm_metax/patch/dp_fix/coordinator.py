# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Note: opt dp internal load balancing
# ---------------------------------------------------------------------------

import time
import msgspec.msgpack
import zmq

from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.serial_utils import MsgpackDecoder

logger = init_logger(__name__)

from vllm.v1.engine.coordinator import EngineState, DPCoordinatorProc


class MacaDPCoordinatorProc(DPCoordinatorProc):
    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
    ):
        decoder = MsgpackDecoder(EngineCoreOutputs)

        # For tracking request wave progression.
        current_wave = 0
        engines_running = False

        # For tracking request counts for internal load-balancing.
        stats_changed = False
        last_stats_step = -1
        last_stats_wave = -1
        last_step_counts: list[list[int]] | None = None

        with (
            make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_front,
            make_zmq_socket(
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as output_back,
            make_zmq_socket(
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_back,
        ):
            # Wait until all engines subscribe.
            for _ in self.engines:
                if publish_back.recv() != b"\x01":
                    logger.error(
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return
            # Send ready message to engines.
            publish_back.send(b"READY")

            logger.info("All engine subscriptions received by DP coordinator")

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0
            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                # Send at stats_update_interval_ms interval if the stats have
                # changed, or otherwise every 5 seconds.
                wait_for = self.stats_update_interval_ms if stats_changed else 5000

                # Wait at least 50ms to ensure we've received all stats for
                # the current step.
                # /------------------------  Metax Modification -------------------------\
                min_timeout = 0  # 50 if last_step_counts is None else 0
                # \------------------------- Metax Modification -------------------------/

                events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
                if not events:
                    # Poller timeout - publish current stats to front-ends.
                    if last_step_counts is not None:
                        engine_req_counts_list = last_step_counts
                        last_step_counts = None
                    else:
                        engine_req_counts_list = self._get_engine_counts()
                        stats_changed = False

                    to_publish = (engine_req_counts_list, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                wave_state_changed = False

                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b"\x01", b"\x00"):
                        # Ignore subscription messages.
                        continue

                    decoded = msgspec.msgpack.decode(buffer)
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 2
                        and decoded[0] == "SCALE_ELASTIC_EP"
                    ):
                        # Handle scale up notification
                        new_engine_count = decoded[1]
                        current_count = len(self.engines)
                        if new_engine_count > current_count:
                            for _ in range(new_engine_count - current_count):
                                self.engines.append(EngineState())
                            # NOTE(yongji): handle the case
                            # where newly started engines have current_wave = 0
                            # if existing engines just finished a wave
                            # and engine_running isn't updated yet at
                            # CoordinatorProc requests routed to newly started
                            # engines may not wake up existing engines, as long
                            # as 0 < request.wave < existing engines'
                            # current_wave
                            # we note that 0 is the wave number for the new
                            # engine
                            engines_running = False
                            logger.info(
                                "DPCoordinator scaled up from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        else:
                            self.engines = self.engines[:new_engine_count]
                            logger.info(
                                "DPCoordinator scaled down from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        continue  # Skip normal engine notification processing

                    # We received a message on the front-end XPUB socket,
                    # from an API server sending a new request while the
                    # engines are paused, so that we can wake the other
                    # engines.
                    engine_to_exclude, wave = decoded
                    if not engines_running:
                        if wave < current_wave:
                            # If the wave number is stale, ensure the message
                            # is handled by all the engines.
                            engine_to_exclude = None

                        engines_running = True
                        wave_state_changed = True
                        self._send_start_wave(
                            publish_back, current_wave, engine_to_exclude
                        )

                if output_back in events:
                    # We received a message from one of the engines.

                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    scheduler_stats = outputs.scheduler_stats
                    if scheduler_stats:
                        # 1. Updated request load stats - update our local
                        # state with these.
                        stats = self.engines[eng_index].request_counts
                        stats_step = scheduler_stats.step_counter
                        stats_wave = scheduler_stats.current_wave
                        if (
                            stats_wave > last_stats_wave
                            or stats_wave == last_stats_wave
                            and stats_step > last_stats_step
                        ):
                            if stats_changed:
                                last_step_counts = self._get_engine_counts(do_copy=True)
                            last_stats_step = stats_step
                            last_stats_wave = stats_wave
                        elif stats_wave != last_stats_wave or (
                            stats_step != last_stats_step
                        ):
                            logger.warning(
                                "Received stats for out-of-order "
                                "step (%d, %d) from engine %d (expected "
                                "> (%d, %d))",
                                stats_wave,
                                stats_step,
                                eng_index,
                                last_stats_wave,
                                last_stats_step,
                            )
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats_changed = True

                    if (wave := outputs.wave_complete) is not None:
                        # 2. Notification from rank 0 engine that we've
                        # moved into the global paused state
                        # (engines_running==False).
                        if current_wave <= wave:
                            new_wave = wave + 1
                            logger.debug(
                                "Moving DP wave from %d to %d.", current_wave, new_wave
                            )
                            current_wave = new_wave
                            engines_running = False
                            wave_state_changed = True
                    elif (wave := outputs.start_wave) is not None and (
                        wave > current_wave
                        or (wave == current_wave and not engines_running)
                    ):
                        # 3. The engine received request for a non-current wave
                        # so we must ensure that other engines progress to the
                        # next wave (race condition handling).
                        logger.debug(
                            "Starting wave %d after notification of "
                            "stale wave request from engine.",
                            wave,
                        )
                        current_wave = wave
                        engines_running = True
                        wave_state_changed = True
                        self._send_start_wave(publish_back, wave, eng_index)

                if wave_state_changed:
                    message = (None, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(message))

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        min_stats_update_interval_ms: int = 100,
    ):
        # /------------------------  Metax Modification -------------------------\
        coordinator = MacaDPCoordinatorProc(
            # \------------------------- Metax Modification -------------------------/
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms,
        )
        try:
            coordinator.process_input_socket(
                front_publish_address,
                back_output_address,
                back_publish_address,
            )
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")


DPCoordinatorProc.process_input_socket = MacaDPCoordinatorProc.process_input_socket
DPCoordinatorProc.run_coordinator = MacaDPCoordinatorProc.run_coordinator
