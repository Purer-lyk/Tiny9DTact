scale = 0.4      # 改这里即可

input_file = "./shape_reconstruction/sensor_obj/black_contact.obj"
output_file = "./shape_reconstruction/sensor_obj/black_contact_scaled.obj"

with open(input_file, "r", encoding="utf-8") as fin:
    lines = fin.readlines()

with open(output_file, "w", encoding="utf-8") as fout:
    for line in lines:
        if line.startswith("v "):
            _, x, y, z = line.split()
            fout.write(
                f"v {float(x)*scale:.6f} {float(y)*scale:.6f} {float(z)*scale:.6f}\n"
            )
        else:
            fout.write(line)

print(f"模型已缩放 {scale} 倍")