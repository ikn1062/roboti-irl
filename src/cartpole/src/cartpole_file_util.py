import csv
import os


def parse_text(cart_path, pole_path, output_path, parse_lines=False):
    cart_file = open(cart_path, 'r')
    pole_file = open(pole_path, 'r')

    cart_lines = cart_file.readlines()
    pole_lines = pole_file.readlines()

    print(f"Length of cart_file: {len(cart_lines)}")
    print(f"Length of pole_file: {len(pole_lines)}")

    if parse_lines:
        output_csv = open(output_path, 'w', newline='')
        output_writer = csv.writer(output_csv)

        for i in range(min(len(cart_lines), len(pole_lines))):
            cart_line = cart_lines[i]
            pole_line = pole_lines[i]

            cart_line_split = cart_line.split(", ")
            pole_line_split = pole_line.split(", ")

            write_line = pole_line_split + cart_line_split

            output_writer.writerow(write_line)

        output_csv.close()

    cart_file.close()
    pole_file.close()


if __name__ == "__main__":
    pass
