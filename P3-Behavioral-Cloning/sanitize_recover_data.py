import csv
import sys

if len(sys.argv) < 3:
  print('Please pass in CSV filename and output filename.')
else:
    fname = sys.argv[1]
    oname = sys.argv[2]
    steering_thresh = 0.1/25

    outfile = open(oname, 'w')
    writer = csv.writer(outfile, delimiter=',')
    ictr = 0
    octr = 0
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if abs(float(row[3])) > steering_thresh:
                writer.writerow(row)
                octr += 1
            ictr += 1

    print('Read ', ictr,' rows')
    print('Wrote ', octr,' rows')
    outfile.close()
