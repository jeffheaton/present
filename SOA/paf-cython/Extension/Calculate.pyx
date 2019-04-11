import math

def sdev(lst):
	# Mean
	sum = 0
	for x in lst:
		sum += x

	mean = sum / len(lst)

	# Standard deviation
	sum = 0
	for x in lst:
		delta = x - mean
		sum += delta ** 2

	return math.sqrt(sum / len(lst))

