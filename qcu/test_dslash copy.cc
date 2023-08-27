#define inverse_clover(input_matrix, inverse_matrix, augmented_matrix, pivot,  \
                       factor)                                                 \
  {                                                                            \
    for (int s0 = 0; s0 < 4; s0++) {                                           \
      for (int c0 = 0; c0 < 3; c0++) {                                         \
        for (int s1 = 0; s1 < 4; s1++) {                                       \
          for (int c1 = 0; c1 < 3; c1++) {                                     \
            inverse_matrix[s0 * 36 + s1 * 9 + c0 * 3 + c1] =                   \
                input_matrix[s0 * 36 + s1 * 9 + c0 * 3 + c1];                  \
            augmented_matrix[s0 * 72 + s1 * 9 + c0 * 3 + c1] =                 \
                inverse_matrix[s0 * 36 + s1 * 9 + c0 * 3 + c1];                \
          }                                                                    \
        }                                                                      \
        augmented_matrix[s0 * 72 + (4 + s0) * 9 + c0 * 3 + c0] = 1.0;          \
      }                                                                        \
    }                                                                          \
    for (int s0 = 0; s0 < 4; s0++) {                                           \
      for (int c0 = 0; c0 < 3; c0++) {                                         \
        pivot = augmented_matrix[s0 * 72 + s0 * 9 + c0 * 3 + c0];              \
        for (int s1 = 0; s1 < 8; s1++) {                                       \
          for (int c1 = 0; c1 < 3; c1++) {                                     \
            augmented_matrix[s0 * 72 + s1 * 9 + c0 * 3 + c1] /= pivot;         \
          }                                                                    \
        }                                                                      \
        for (int s1 = 0; s1 < 4; s1++) {                                       \
          for (int c1 = 0; c1 < 3; c1++) {                                     \
            if ((s0 != s1) || (c0 != c1)) {                                    \
              factor = augmented_matrix[s1 * 72 + s0 * 9 + c1 * 3 + c0];       \
              for (int ss1 = 0; ss1 < 8; ss1 +) {                              \
                for (int cc1 = 0; cc1 < 3; cc1++) {                            \
                  augmented_matrix[s1 * 72 + ss1 * 9 + c1 * 3 + cc1] -=        \
                      factor *                                                 \
                      augmented_matrix[s0 * 72 + ss1 * 9 + c0 * 3 + cc1];      \
                }                                                              \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    for (int s0 = 0; s0 < 4; s0++) {                                           \
      for (int c0 = 0; c0 < 3; c0++) {                                         \
        for (int s1 = 0; s1 < 4; s1++) {                                       \
          for (int c1 = 0; c1 < 3; c1++) {                                     \
            inverse_matrix[s0 * 36 + s1 * 9 + c0 * 3 + c1] =                   \
                augmented_matrix[s0 * 72 + (4 + s1) * 9 + c0 * 3 + c1];        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }
