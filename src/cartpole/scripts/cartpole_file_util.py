#!/usr/bin/env python3
import csv
import sys


def parse_text(cart_path, pole_path, output_path, parse_lines=True):
    """
    Parse Text is a function used to combine simulation demonstrations from cart and pole txt files into a csv output

    Args:
        cart_path (str): Path to cart txt file
        pole_path (str): Path to pole txt file
        output_path (str): Path to output csv file
        parse_lines (bool): Bool to enable line parser and output 

    Returns:
        None
    """
    cart_file = open(cart_path, 'r')
    pole_file = open(pole_path, 'r')

    cart_lines = cart_file.readlines()
    pole_lines = pole_file.readlines()

    if parse_lines:
        output_csv = open(output_path, 'w', newline='')
        output_writer = csv.writer(output_csv)
        
        for i in range(min(len(cart_lines), len(pole_lines))):
            cart_line = cart_lines[i].strip("\n")
            pole_line = pole_lines[i].strip("\n")

            cart_line_split = cart_line.split(", ")
            pole_line_split = pole_line.split(", ")

            write_line = pole_line_split + cart_line_split

            output_writer.writerow(write_line)

        output_csv.close()

    cart_file.close()
    pole_file.close()

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise TypeError("Wrong number of parameters inputted for File Utility")
    cart_file_path, pole_file_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    parse_text(cart_file_path, pole_file_path, output_path)

    
