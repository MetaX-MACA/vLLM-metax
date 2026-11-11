# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Case loading, suite execution and scoring; no resource or process ownership."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import yaml

from tools.batched_test.api_client import (
    ApiConnection,
    ChatCompletionClient,
    EmbeddingClient,
    cosine_similarity,
)
from tools.batched_test.deployment import CRITICAL_WORDS
from tools.batched_test.results import CaseResult
from tools.batched_test.specs import (
    CaseSpec,
    InferenceSpec,
    ModelSpec,
    SuiteSpec,
    positive_int,
)
from tools.batched_test.paths import TOOL_ROOT, is_url, resolve_path


def load_cases(
    path: str | None, kind: str, default_max_tokens=256
) -> tuple[CaseSpec, ...]:
    if not path:
        raise ValueError(f"Case file required for {kind}")
    source = Path(resolve_path(path, TOOL_ROOT))
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Case file must contain a nonempty list: {path}")
    cases = []
    key = {
        "text-only": "question",
        "single-image": "picture_url",
        "embedding": "query",
    }[kind]
    for row in data:
        if not isinstance(row, dict) or not isinstance(row.get(key), str):
            raise ValueError(f"Every {kind} case needs a string {key}: {path}")
        keywords = row.get("keywords", [])
        positives, negatives = row.get("positive", []), row.get("negative", [])
        if not all(
            isinstance(items, list) for items in (keywords, positives, negatives)
        ):
            raise ValueError(f"keywords/positive/negative must be lists: {path}")
        if kind == "embedding" and (not positives or not negatives):
            raise ValueError(
                f"Embedding cases need positive and negative texts: {path}"
            )
        case_input = row[key]
        if kind == "single-image" and not is_url(case_input):
            case_input = resolve_path(case_input, source.parent)
        cases.append(
            CaseSpec(
                case_input,
                tuple(map(str, keywords)),
                positive_int(row.get("max_tokens", default_max_tokens), "max_tokens"),
                tuple(map(str, positives)),
                tuple(map(str, negatives)),
            )
        )
    return tuple(cases)


def plan_inference(
    model: ModelSpec,
    *,
    text_case: str | None,
    image_case: str | None,
    embedding_case: str | None = None,
    long_text_case: str | None = None,
    validation_mode: str | None = None,
) -> InferenceSpec:
    if not model.infer_types:
        raise ValueError(f"{model.name}: infer_type must be specified")
    paths = {
        "text-only": text_case,
        "single-image": image_case,
        "embedding": embedding_case,
    }
    unsupported = set(model.infer_types) - paths.keys()
    if unsupported:
        raise ValueError(f"{model.name}: unsupported infer_type: {sorted(unsupported)}")
    suites = []
    for kind in model.infer_types:
        cases = load_cases(paths[kind], kind)
        if kind == "text-only" and long_text_case:
            long_cases = load_cases(long_text_case, kind, default_max_tokens=1024)
            # Stable across queue order and port allocation, unlike port % N.
            index = int(hashlib.sha256(model.name.encode()).hexdigest(), 16) % len(
                long_cases
            )
            cases += (long_cases[index],)
        suites.append(SuiteSpec(kind, cases))
    mode = validation_mode or (
        "dummy_smoke" if model.serve.load_format == "dummy" else "accuracy"
    )
    if (mode == "dummy_smoke") != (model.serve.load_format == "dummy"):
        raise ValueError(
            "Dummy weights require dummy_smoke; real weights require accuracy"
        )
    return InferenceSpec(tuple(suites), validation_mode=mode)


def score_keywords(content: str, keywords: tuple[str, ...]) -> bool:
    for word in CRITICAL_WORDS:
        if word in content:
            raise RuntimeError(f"Client received critical server error: {word}")
    return any(key.lower() in content.lower() for key in keywords)


def run_suites(context, spec: InferenceSpec, port: int) -> tuple[CaseResult, ...]:
    if spec.validation_mode == "dummy_smoke":
        return run_dummy_smoke(context, spec, port)
    results = []
    connection = ApiConnection(
        port=port, stop_event=context.cancel, timeout=context.request_timeout
    )
    try:
        chat = ChatCompletionClient(connection=connection)
        embedding = EmbeddingClient(connection=connection)
        for suite in spec.suites:
            log = context.work_dir / f"{suite.kind}_inference.log"
            with log.open("a", encoding="utf-8") as stream:
                for index, case in enumerate(suite.cases):
                    context.check()
                    if suite.kind == "embedding":
                        vectors = embedding.embed(
                            [case.input, *case.positive, *case.negative]
                        )
                        if len(vectors) != 1 + len(case.positive) + len(case.negative):
                            raise RuntimeError(
                                "Embedding response count does not match input"
                            )
                        sims = [cosine_similarity(vectors[0], v) for v in vectors[1:]]
                        pos = sum(sims[: len(case.positive)]) / len(case.positive)
                        neg = sum(sims[len(case.positive) :]) / len(case.negative)
                        content = f"mean_pos_sim={pos:.4f} mean_neg_sim={neg:.4f}"
                        passed = pos > neg
                    elif suite.kind == "single-image":
                        content = next(
                            chat.run_single_image(case.max_tokens, [case.input])
                        )
                        passed = score_keywords(content, case.keywords)
                    else:
                        content = next(
                            chat.run_text_only(
                                [case.input], max_completion_tokens=case.max_tokens
                            )
                        )
                        passed = score_keywords(content, case.keywords)
                    context.check()
                    result = CaseResult(
                        suite.kind,
                        index,
                        passed,
                        content,
                        "" if passed else "Answer did not satisfy case criteria",
                    )
                    results.append(result)
                    stream.write(
                        f"Input: {case.input[:2000]}\nResponse: {content}\nPassed: {passed}\n"
                    )
    finally:
        connection.close()
    return tuple(results)


def run_dummy_smoke(context, spec, port):
    """Concurrent requests check execution and decode, without semantic claims."""
    connection = ApiConnection(
        port=port, stop_event=context.cancel, timeout=context.request_timeout
    )
    jobs = [
        (suite.kind, index, case)
        for suite in spec.suites
        for index, case in enumerate(suite.cases)
    ]
    if len(jobs) == 1:
        kind, index, case = jobs[0]
        jobs.append((kind, index + 1, case))

    def execute(job):
        kind, index, case = job
        context.check()
        if kind == "embedding":
            vectors = EmbeddingClient(connection=connection).embed([case.input])
            passed = (
                len(vectors) == 1
                and bool(vectors[0])
                and all(math.isfinite(value) for value in vectors[0])
            )
            content = f"embedding_dimension={len(vectors[0]) if vectors else 0}"
        else:
            client = ChatCompletionClient(connection=connection)
            maximum = max(2, case.max_tokens)
            if kind == "single-image":
                content = next(
                    client.run_single_image(maximum, [case.input], smoke=True)
                )
            else:
                content = next(
                    client.run_text_only(
                        [case.input], max_completion_tokens=maximum, smoke=True
                    )
                )
            passed = bool(content.strip()) and client.last_completion_tokens >= 2
            if any(word in content for word in CRITICAL_WORDS):
                passed = False
            completion_tokens = client.last_completion_tokens
        if kind == "embedding":
            completion_tokens = None
        context.check()
        return CaseResult(
            kind,
            index,
            passed,
            content,
            ""
            if passed
            else "Dummy execution smoke failed; no semantic accuracy claim",
            completion_tokens,
        )

    try:
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
            results = tuple(pool.map(execute, jobs))
        with (context.work_dir / "dummy_smoke.log").open("w") as stream:
            for result in results:
                stream.write(
                    f"{result.suite}:{result.index} passed={result.passed} "
                    f"completion_tokens={result.completion_tokens} "
                    f"response={result.response}\n"
                )
        return results
    finally:
        connection.close()
