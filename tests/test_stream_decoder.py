from ptu_decoder.ptu_decoder import StreamDecoder

with open("test_data.txt", "r") as f:
    data = f.read().splitlines()

d = StreamDecoder()

for da in data:
    i = int(da, 2)
    d.decode(i)

# print(d.records)
print(d.num_records)


# with open("test_output.txt", "r") as f:
#     data = f.read().splitlines()
# print(len([l for l in data if "CHN 0 " in l or "OFL * " in l]))
