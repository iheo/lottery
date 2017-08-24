import numpy as np

bucket = np.zeros(69, dtype=np.int32)
bucket_pb = np.zeros(26, dtype=np.int32)

with open('winnums-text.txt', 'r') as f:
    for i, line in enumerate(f):
        if i > 0:
            one_line = line.split()
            nums= np.array(one_line[1:6], dtype=np.int32)
            for num in nums:
                bucket[num-1] += 1
            bucket_pb[np.int(one_line[6])-1] += 1


def sum_to_unit_range(nums):
    cumsum_bucket = np.cumsum(nums, dtype=np.float32)
    pr_bucket = cumsum_bucket/cumsum_bucket[-1]
    return pr_bucket

pr_bucket = sum_to_unit_range(bucket)
pr_bucket_pb = sum_to_unit_range(bucket_pb)

N = 100
num = []
num_pb = []
np.random.seed(20170823)
# np.random.seed(19810711)
# np.random.seed()
for k in xrange(N):
    rand = np.random.uniform(0, 1, 1)
    draw = np.min(np.where(rand < pr_bucket))+1
    num.append(draw)
    if len(np.unique(num)) == 5:
        break
    # np.random.seed(draw)

# Power ball
pb_bucket = np.zeros(26)
rand = np.random.uniform(0, 1, 1)
num_pb = np.min(np.where(rand < pr_bucket_pb)) + 1

print num
print num_pb
