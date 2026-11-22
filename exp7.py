import cv2
import heapq
from collections import Counter

img = cv2.imread("image.jpg",0)
if img is None: raise FileNotFoundError("Put image.jpg in this folder")
img = cv2.resize(img,(100,100))
a = img.flatten().tolist()
original = len(a)

# 1. Run Length Encoding (RLE)
runs = []
for value in a:
    if runs and runs[-1][0]==value:
        runs[-1][1] += 1
    else:
        runs.append([value,1])
rle_size = len(runs)*5   # 1 byte value + 4 bytes count
rle_decoded = [v for value,count in runs for v in [value]*count]

# 2. Huffman coding using a min heap
freq = Counter(a)
heap = [[count,[value,""]] for value,count in freq.items()]
heapq.heapify(heap)
while len(heap)>1:
    left = heapq.heappop(heap)
    right = heapq.heappop(heap)
    for pair in left[1:]: pair[1] = '0'+pair[1]
    for pair in right[1:]: pair[1] = '1'+pair[1]
    heapq.heappush(heap,[left[0]+right[0]]+left[1:]+right[1:])
codes = {value: code or '0' for value,code in heap[0][1:]}
huffman_bits = sum(freq[v]*len(codes[v]) for v in freq)
huffman_size = (huffman_bits+7)//8 + 2*len(freq)

# 3. LZW dictionary encoding
# Quantize to 16 gray levels for repeating patterns, as in the PDF
symbols = (img//16).flatten().tolist()
dictionary = {(i,):i for i in range(256)}
next_code = 256
sequence = ()
lzw_codes = []
for value in symbols:
    candidate = sequence+(value,)
    if candidate in dictionary:
        sequence = candidate
    else:
        lzw_codes.append(dictionary[sequence])
        dictionary[candidate] = next_code
        next_code += 1
        sequence = (value,)
if sequence: lzw_codes.append(dictionary[sequence])
lzw_size = len(lzw_codes)*2  # estimated 2 bytes per code

print("Original:",original,"bytes")
print("RLE:",rle_size,"bytes, lossless:",rle_decoded==a)
print("Huffman:",huffman_size,"bytes")
print("LZW (16-level image):",lzw_size,"bytes")
cv2.imshow("Original",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
