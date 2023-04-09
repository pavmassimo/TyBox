import serial
import time
import sys
import threading

import time
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

test_set_size = 100
test_interval = 50


class myThread (threading.Thread):
    def __init__(self, res):
        threading.Thread.__init__(self)
        self.res = res
    def run(self):
       fig = plt.figure()
       ax = fig.add_subplot(1, 1, 1)
       xs = []
       ys = []

       datoAggiornato = self.res.get_last()
       # print("got data point")
       xs.append(datoAggiornato[1])
       ys.append(datoAggiornato[0])

       ani = animation.FuncAnimation(fig, animate, fargs=(self.res, ax, xs, ys), interval=1000)
       plt.show()


def animate(i, res, ax, xs, ys):
    # print(res.c)
    datoAggiornato = res.get_last()
    # print("got data point")
    xs.append(datoAggiornato[1])
    ys.append(datoAggiornato[0])
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    # plt.xticks([])
    # plt.subplots_adjust(bottom=0.30)
    plt.title('accuracy vs training samples')
    plt.ylabel('accuracy')
    plt.xlabel('samples')



class Results:
    def __init__(self):
        self.counts = [0.12]
        self.values = [0]

    def add_point(self, count, value):
        self.counts.append(count)
        self.values.append(value)

    def get_last(self):
        return self.counts[-1], self.values[-1]


if __name__ == "__main__":
    address = sys.argv[1]
    file_path = sys.argv[2]
    baud_rate = sys.argv[3]

    r = Results()
    t = myThread(r)
    t.start()

    try:
        port = serial.Serial(address, baud_rate)
    except Exception as e:
        print("error\n", repr(e), "\nexiting...")
        sys.exit(1)
    print("Port open")
    print(port.readline())
    print("start experiment")
    with open(file_path, "r") as file:
        if not file:
            print("could not open file. exiting...")
            sys.exit(2)
        print("file open")
        line = file.readline()
        line_number = 0
        while line != '':
            port.write(line.encode())
            line_number += 1
            # print(port.readline())
            # if line_number % 10 == 0:
            print(line_number)
            if line_number % test_interval == 0:
                test_line_number = 0
                with open("data/transfer_fashion_test.data", "r") as test_file:
                    line_test = test_file.readline()
                    print("testing...", line_number)
                    while line_test != '':
                        if test_line_number >= test_set_size:
                            break
                        print("test line", test_line_number)
                        port.write(line_test.encode())
                        # print(port.readline())
                        line_test = test_file.readline()
                        test_line_number += 1
                reading = port.readline()
                # print(reading, float(reading[7:13]), int(reading[22:24]))
                print(reading)
                r.add_point(float(reading[7:13]), line_number)
            line = file.readline()
        print("Finished streaming file. Exiting")
    port.write('r\n'.encode())

    port.close()
