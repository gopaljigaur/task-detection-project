

class DescriptorConfiguration:

    precision = None
    recall = None
    f1 = None
    tp = None
    fn = None
    fp = None
    tn = None
    multiple = None
    total = None

    def __init__(self, descriptor_set, threshold, aggregation_percentage):
        self.descriptor_set = descriptor_set
        self.threshold = threshold
        self.aggregation_percentage = aggregation_percentage

    def record_performance(self, true_positive, false_negative, false_positive, true_negative, multiple, total):
        self.tp = true_positive
        self.fn = false_negative
        self.fp = false_positive
        self.tn = true_negative
        self.multiple = multiple
        self.total = total
        self.precision = self._precision()
        self.recall = self._recall()
        self.f1 = self._f1()

    def _precision(self):
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def _recall(self):
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def _f1(self):
        if self.precision + self.recall == 0:
            return 0
        return 2 * ((self.precision * self.recall) / (self.precision + self.recall))
