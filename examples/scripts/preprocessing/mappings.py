
class MiccaiMapping(object):
    def __init__(self):
        self.all_labels = [0, 4, 11, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40,
                           41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55,
                           56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 69, 71, 72,
                           73, 75, 76, 100, 101, 102, 103, 104, 105, 106, 107,
                           108, 109, 112, 113, 114, 115, 116, 117, 118, 119,
                           120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                           132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
                           142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
                           152, 153, 154, 155, 156, 157, 160, 161, 162, 163,
                           164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
                           174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                           184, 185, 186, 187, 190, 191, 192, 193, 194, 195,
                           196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
                           206, 207]
        self.ignore_labels = [1, 2, 3] + \
                             [5, 6, 7, 8, 9, 10] + \
                             [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] + \
                             [24, 25, 26, 27, 28, 29] + \
                             [33, 34] + [42, 43] + [53, 54] + \
                             [63, 64, 65, 66, 67, 68] + [70, 74] + \
                             [80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                              90, 91, 92, 93, 94, 95, 96, 97, 98, 99] + \
                             [110, 111, 126, 127, 130, 131, 158, 159, 188, 189]
        self.overall_labels = set(self.all_labels).difference(
            set(self.ignore_labels))

        self.cortical_labels = [x for x in self.overall_labels if x >= 100]
        self.non_cortical_labels = \
            [x for x in self.overall_labels if x > 0 and x < 100]

        self.map = {v: k for k, v in enumerate(self.overall_labels)}
        self.reversed_map = {k: v for k, v in enumerate(self.overall_labels)}
        self.nb_classes = len(self.overall_labels)

    def __getitem__(self, index):
        return self.map[index]


class OASISMapping(object):
    def __init__(self):
        self.avoid_train_patients = [
            '0061', '0080', '0092', '0145', '0150', '0156', '0191', '0202',
            '0230', '0236', '0239', '0249', '0285', '0353', '0368'
        ]

        self.avoid_test_patients = [
            '0101', '0111', '0117', '0379', '0395', '0101', '0111', '0117',
            '0379', '0395', '0091', '0417', '0040', '0282', '0331', '0456',
            '0300', '0220', '0113', '0083'
        ]

        self.avoid_patients = self.avoid_train_patients + self.avoid_train_patients


class IbsrMapping(object):
    def __init__(self):
        self.all_labels = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17,
                           18, 24, 26, 28, 29, 30, 41, 42, 43, 44, 46, 47, 48,
                           49, 50, 51, 52, 53, 54, 58, 60, 61, 62, 72]

        # structures : undetermined (29,61), vessel(72,30), 5th ventricule (62)
        self.ignore_labels = [29, 61, 72, 30, 62]
        self.overall_labels = sorted(list(set(self.all_labels).difference(
            set(self.ignore_labels))))

        self.map = {}
        index = 0
        for v in self.overall_labels:
            if v == 49:
                self.map.update({49: self.map[48]})
            else:
                self.map.update({v: index})
                index += 1

        self.reversed_map = {k: v for k, v in enumerate(self.overall_labels)}
        self.nb_classes = len(self.overall_labels) - 1

    def __getitem__(self, index):
        return self.map[index]
