import numpy as np


class ArrayInitialization:
    def array_check(self, arr):
        if type(arr) == list:
            self.new_arr = np.array(arr, dtype=float)
        elif type(arr) == np.ndarray:
            if len(arr.shape) == 1:
                self.new_arr = np.array(arr, dtype=float)
            else:
                if arr.shape[0] < arr.shape[1]:
                    self.new_arr = np.squeeze(arr, axis=0)
                else:
                    self.new_arr = np.squeeze(arr, axis=1)
                self.new_arr = np.array(self.new_arr, dtype=float)

        return self.new_arr

    def gen_init_arrays(self, t0, y0):
        self.ysol = np.empty(0)
        self.yhatsol = np.empty(0)
        self.tsol = np.empty(0)

        self.ysol = np.append(self.ysol, y0)
        self.yhatsol = np.append(self.yhatsol, y0)
        self.tsol = np.append(self.tsol, t0)

        return self.tsol, self.ysol, self.yhatsol


###----------------------------------------###
