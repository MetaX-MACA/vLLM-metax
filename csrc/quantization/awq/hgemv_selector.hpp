// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <memory>
#include <vector>
#include <algorithm>
#include "mc_runtime.h"
#include "maca_fp16.h"

struct KernelEventRecorder {
    mcEvent_t _start;
    mcEvent_t _stop;
    float _eventMs = -1.f;

    KernelEventRecorder() {
        mcEventCreate(&_start);
        mcEventCreate(&_stop);
    }

    ~KernelEventRecorder() {
        mcEventDestroy(_start);
        mcEventDestroy(_stop);
    }

    void start() {
        mcEventRecord(_start, NULL);
    }

    float stop() {
        mcEventRecord(_stop, NULL);
        mcEventSynchronize(_stop);
        mcEventElapsedTime(&_eventMs, _start, _stop);
        return _eventMs;
    }
};
namespace hgemv_selector {
template<int quant_group=128, int pack_ratio=8, int m_per_thread=4>
struct GemvParamAutoSelector {
    int m;
    int k;
    int best_block_x = 0;
    int best_split_k = 0;

    std::pair<int,int> block_x_range;
    std::pair<int,int> split_k_range;
    bool _valid = false;

private:
    std::vector<std::pair<int, int>> param_candidates;
    int warmup_iters = 0;
    int current_block_x = 0;
    int current_split_k = 0;
    int current_perf_iter = 0;
    std::vector<float> perf_times;
    float kernel_best_time_ms_ave = 99999999.0f;
    float best_band_width;
    float data_size_gb;
    std::shared_ptr<KernelEventRecorder> _r;
    bool _selected = false;
    const static int MAX_PERF_COUNT = 20;

public:
    GemvParamAutoSelector(int _m, int _n, int _k, bool allow_imcomplete_bx = false) : m(_m), k(_k)
    {
        if (!prepare_parameters(_m, _n, _k, block_x_range.second, split_k_range.second, allow_imcomplete_bx)) return;
        block_x_range.first = block_x_range.second;
        split_k_range.first = split_k_range.second;
        for (int i = 0; i < 3; i++) if (block_x_range.first / 2 >= 16) block_x_range.first /= 2;
        for (int i = 0; i < 3; i++) if (split_k_range.first % 2 == 0 && split_k_range.first / 2 > 0) split_k_range.first /= 2;
        for (int i = block_x_range.first; i <= block_x_range.second; i *= 2) {
            for (int j = split_k_range.first; j <= split_k_range.second; j *= 2) {
                param_candidates.emplace_back(i, j);
            }
        }
        if (split_k_range.second * quant_group != k) {
            int max_split_k = k / quant_group;
            if (max_split_k < 256) {
                for (int i = block_x_range.first; i <= block_x_range.second; i *= 2) {
                    param_candidates.emplace_back(i, max_split_k);
                }
            }
        }

        current_block_x = block_x_range.second;
        current_split_k = split_k_range.second;
        uint64_t data_size_in_bytes = _n * _k * sizeof(half);
        data_size_in_bytes += (uint64_t)_m * _k / pack_ratio * sizeof(uint32_t);
        data_size_in_bytes += (uint64_t)_m * _k / quant_group * sizeof(half);
        data_size_in_bytes += (uint64_t)_m * _k / quant_group / pack_ratio * sizeof(uint32_t);
        data_size_in_bytes += (uint64_t)_n * _m * sizeof(half);
        data_size_gb = float(data_size_in_bytes) / 1024 / 1024 / 1024;
        warmup_iters = 4;
        _valid = true;
    }

    void select_in_warmup(const std::function<bool(int,int)> &f) {
        if (!valid()) {
            printf("Cannot run this selector!\n");
            return;
        }
        if (_selected) {
            f(best_block_x, best_split_k);
            return;
        };
        //Warmup
        for (auto iter = param_candidates.begin(); iter != param_candidates.end(); iter++) {
            for (int i = 0; i < 5; i++) {
                auto &p = *iter;
                f(p.first, p.second);
            }
        }
        _r.reset(new KernelEventRecorder());
        kernel_best_time_ms_ave = 9999999.0f;
        mcDeviceSynchronize();

        for (auto iter = param_candidates.begin(); iter != param_candidates.end(); iter++) {
            auto &p = *iter;
            auto &bx = p.first;
            auto &sk = p.second;
            mcDeviceSynchronize();
            _r->start();
            bool launched = false;
            for (int i = 0; i < MAX_PERF_COUNT; i++) {
                launched = f(bx, sk);
            }
            auto ms = _r->stop();
            if (launched && ms / MAX_PERF_COUNT < kernel_best_time_ms_ave) {
                best_block_x = bx;
                best_split_k = sk;
                kernel_best_time_ms_ave = ms / MAX_PERF_COUNT;
            }
        }

        _r.reset();
        _selected = true;
        warmup_iters = 0;
    }

    void run(const std::function<bool(int,int)> &f) {
        if (!valid()) {
            printf("Cannot run this selector!\n");
            return;
        }
        if (warmup_iters > 0) {
            f(current_block_x, current_split_k);
            //printf("Calling warmup current_block_x = %d, current_split_k = %d\n", current_block_x, current_split_k);
            if (current_block_x > block_x_range.first) current_block_x /= 2;
            else if (current_split_k > split_k_range.first) current_split_k /= 2;
            if (current_block_x == block_x_range.first && current_split_k >= split_k_range.first) {
                current_block_x = block_x_range.second;
                current_split_k = split_k_range.second;
            }
            warmup_iters--;
            mcDeviceSynchronize();
            return;
        }

        if (_selected) {
            f(best_block_x, best_split_k);
        } else {
            if (!_r) {
                _r = std::shared_ptr<KernelEventRecorder>(new KernelEventRecorder());
                current_block_x = block_x_range.second;
                current_split_k = split_k_range.second;
                current_perf_iter = MAX_PERF_COUNT;
            }
            _r->start();
            auto launched = f(current_block_x, current_split_k);
            auto ms = _r->stop();
            //printf("Launched with bx=%d,sk=%d,iter=%d,ms=%fms\n", current_block_x, current_split_k, current_perf_iter, ms);
            if (!launched) {
                printf("ERROR: Cannot run gemv with bx = %d, sk = %d\n", current_block_x, current_split_k);
                return;
            }
            if (current_perf_iter-- > 0) {
                perf_times.emplace_back(ms);
                return;
            }

            std::sort(perf_times.begin(), perf_times.end());
            float total_tm = 0;
            for (size_t i = 1; i < perf_times.size() - 1; i++) {
                total_tm += perf_times[i];
            }

            ms = total_tm /= (MAX_PERF_COUNT - 2);
            perf_times.clear();
            current_perf_iter = MAX_PERF_COUNT;
            //printf("get ave time %fms\n", ms);

            if (ms < kernel_best_time_ms_ave) {
                best_block_x = current_block_x;
                best_split_k = current_split_k;
                kernel_best_time_ms_ave = ms;
            }

            if (current_split_k > split_k_range.first) {
                current_split_k /= 2;
            } else if (current_block_x > block_x_range.first){
                current_split_k = split_k_range.second;
                current_block_x /= 2;
            } else if (current_block_x == block_x_range.first && current_split_k == split_k_range.first) {
                _selected = true;
                _r.reset();
            } else {
                printf("Error: all parameters are tried! current_block_x = %d, current_split_k = %d, block_x_range = %d,%d, split_k_range = %d,%d, best_block_x = %d, best_split_k = %d\n",
                    current_block_x, current_split_k,
                    block_x_range.first, block_x_range.second,
                    split_k_range.first, split_k_range.second,
                    best_block_x, best_split_k
                );
            }
        }
    }

    bool valid() const { return _valid; }
    bool selected() const {return _selected; }

    float gemv_ave_time_us_cost() {
        return kernel_best_time_ms_ave;
    }

    float gemv_bandwidth() {
        return data_size_gb / (kernel_best_time_ms_ave / 1000.0f);
    }

    static bool prepare_parameters(int m, int n, int k, int &bx, int &sk, bool allow_imcomplete_bx = false) {
        if (n > 4) return false;
        if (k % quant_group != 0) return false;
        if (m < 16 * m_per_thread) return false;
        int max_split_k = k / quant_group;
        int proper_splitk = 1;
        while (proper_splitk*2 <= 256 && proper_splitk*2 <= max_split_k) proper_splitk *= 2;

        int thread_block = 256;  //A thread block of 512 will wait for more warps when doing syncthreads

        int proper_bx = 16;
        if (m % (proper_bx * m_per_thread) != 0) return false;
        //proper_bx * m_per_threads corresponds to group_elements, this value should not be larger than 2048 as limit of shared memory
        while (proper_bx * 2 * m_per_thread && (m % (proper_bx*2*m_per_thread) == 0) && proper_bx * 2 <= 256 && proper_bx * 2 * m_per_thread <= 2048) proper_bx*=2;
        if (allow_imcomplete_bx) {
            int may_proper_bx = proper_bx * 2;
            if (may_proper_bx <= 256 && may_proper_bx * m_per_thread <= 2048 && (m % (may_proper_bx*m_per_thread)) > may_proper_bx * m_per_thread / 4) {
                proper_bx = may_proper_bx;
            }
        }

        bx = proper_bx;
        sk = proper_splitk;
        //printf("m=%d,n=%d,k=%d,get bx=%d,sk=%d,allow_imcomplete_bx=%d\n", m, n, k, bx, sk, allow_imcomplete_bx);
        return true;
    }
};

template<int quant_group=128, int pack_ratio=8, int m_per_thread=4>
class GemvSelectorHolder {
private:
    std::vector<GemvParamAutoSelector<quant_group,pack_ratio,m_per_thread>> _selectors;
    static std::shared_ptr<GemvSelectorHolder<quant_group,pack_ratio,m_per_thread>> _holder;
    static GemvParamAutoSelector<quant_group,pack_ratio,m_per_thread> _invalid_selector;

public:
    static GemvParamAutoSelector<quant_group,pack_ratio,m_per_thread>& selector(int m, int n, int k, bool allow_imcomplete_bx = false) {
        if (!GemvSelectorHolder::_holder) {
            GemvSelectorHolder::_holder.reset(new GemvSelectorHolder<quant_group,pack_ratio,m_per_thread>());
        }
        int bx, sk;
        if (!GemvParamAutoSelector<quant_group,pack_ratio,m_per_thread>::prepare_parameters(m, n, k, bx, sk, allow_imcomplete_bx)) {
            return _invalid_selector;
        }
        auto iter = std::find_if(_holder->_selectors.begin(), _holder->_selectors.end(),
            [&](const GemvParamAutoSelector<quant_group,pack_ratio,m_per_thread>& p) -> bool {
                return p.m == m && p.k == k;
            });
        if (iter != _holder->_selectors.end()) return *iter;
        GemvParamAutoSelector<quant_group,pack_ratio,m_per_thread> sl(m, n, k, allow_imcomplete_bx);
        if (!sl.valid()) {
            return _invalid_selector;
        }
        _holder->_selectors.emplace_back(sl);
        return _holder->_selectors.back();
    }
};

template<int quant_group, int pack_ratio, int m_per_thread>
std::shared_ptr<GemvSelectorHolder<quant_group,pack_ratio,m_per_thread>> GemvSelectorHolder<quant_group,pack_ratio,m_per_thread>::_holder;

template<int quant_group, int pack_ratio, int m_per_thread>
GemvParamAutoSelector<quant_group,pack_ratio,m_per_thread> GemvSelectorHolder<quant_group,pack_ratio,m_per_thread>::_invalid_selector(0,8,0);
}