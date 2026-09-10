// RAS adapter only. FDE arithmetic lives in Google's unmodified source.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>
#include "sketching/point_cloud/fixed_dimensional_encoding.h"

extern "C" int fde_encode(const float* values, const int64_t* offsets,
    int64_t count, int dim, int repetitions, int bits, int projection,
    int final_dimension, int seed, int query, int threads, float* output,
    char* error, int error_size) {
  if (count < 1 || dim < 1 || repetitions < 1 || bits < 0 || bits > 12 ||
      projection < 0 || final_dimension < 0 || seed < 0 || threads < 1) return 1;
  const int width = final_dimension ? final_dimension :
      repetitions * (1 << bits) * (projection ? projection : dim);
  graph_mining::FixedDimensionalEncodingConfig config;
  config.set_dimension(dim);
  config.set_num_repetitions(repetitions);
  config.set_num_simhash_projections(bits);
  config.set_seed(seed);
  config.set_fill_empty_partitions(!query);
  config.set_encoding_type(query ? config.DEFAULT_SUM : config.AVERAGE);
  if (projection) {
    config.set_projection_type(config.AMS_SKETCH);
    config.set_projection_dimension(projection);
  }
  if (final_dimension) config.set_final_projection_dimension(final_dimension);
  int failed = 0;
  #pragma omp parallel for num_threads(threads) schedule(dynamic) reduction(|:failed)
  for (int64_t i = 0; i < count; ++i) {
    try {
      if (offsets[i+1] <= offsets[i]) { failed = 1; continue; }
      std::vector<float> cloud(values + offsets[i]*dim, values + offsets[i+1]*dim);
      auto result = graph_mining::GenerateFixedDimensionalEncoding(cloud, config);
      if (!result.ok()) {
        #pragma omp critical
        std::snprintf(error, error_size, "%s", result.status().ToString().c_str());
        failed = 1;
      } else if (result->size() != static_cast<size_t>(width)) {
        failed = 1;
      } else {
        std::copy(result->begin(), result->end(), output + i*width);
      }
    } catch (const std::exception& e) {
      #pragma omp critical
      std::snprintf(error, error_size, "%s", e.what());
      failed = 1;
    }
  }
  return failed;
}
