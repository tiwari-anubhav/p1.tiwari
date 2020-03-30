import psutil
import energyusage
import os
import cProfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import random
from KNNClassification import KNN
from SVMClassification import SVM


def print_stats(p1, start_time):
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    # print('Process ID:' + str(p1) + ', End Time:' + str(end_time) + '\n')
    print('-' * 30)
    print("USER LEVEL INFORMATION")
    print('# Time Taken: ' + str(time_taken) + ' seconds')
    print('-' * 30)
    print("SYSTEM LEVEL INFORMATION")
    py = psutil.Process(p1)
    memory_use = py.memory_info()[0]/2.**30
    full_memory = py.memory_full_info()
    print('# CPU Cores: \n', py.cpu_num())
    print('# CPU Threads for Process: \n', py.num_threads())
    print('# CPU Usage: User, System, Interrupt \n', py.cpu_times())
    print('# Memory use: \n', memory_use, "GB")
    print('# Hard drive usage: \n', psutil.disk_usage('/home/anubhav/PycharmProjects/AOS/Project1/main.py').percent)
    print('# RSS: \n', full_memory[0])
    print('# VMS: \n', full_memory[1])
    print('# CPU Utilization \n', psutil.cpu_percent())
    print('# Context Switches \n', py.num_ctx_switches())
    print('# Memory Analysis: \n', full_memory)
    print('# Memory Percent: \n', py.memory_percent())


if __name__ == '__main__':
    scenario = input("Enter 1 for executing first program, 2 for second and 3 for concurrent execution: ")
    scenario = int(scenario)
    if scenario == 1:
        k = input("Enter your value of K between 1 to 10: ")
        k = int(k)
        print('Printing Stats for Algorithm KNN alone')
        start = datetime.now()
        print('Process ID: ' + str(os.getpid()) + ', Start Time: ' + str(start))
        knn = KNN()
        program = knn.knn_predict(k)
        pid = os.getpid()
        print_stats(pid, start)

    elif scenario == 2:
        g = input("Enter your value of gamma between 0 and 1: ")
        g = float(g)
        print('Printing Stats for Algorithm SVM alone')
        start = datetime.now()
        print('Process ID: ' + str(os.getpid()) + ', Start Time: ' + str(start))
        svm = SVM()
        program = svm.svm_predict(g)
        pid = os.getpid()
        print_stats(pid, start)

    elif scenario == 3:
        k = input("Enter your value of K between 1 to 10: ")
        k = int(k)
        g = input("Enter your value of gamma between 0 and 1: ")
        g = float(g)
        print('Printing Stats for Algorithm KNN and SVM running Concurrently')
        start = datetime.now()
        print('Process ID: ' + str(os.getpid()) + ', Start Time: ' + str(start))
        knn = KNN()
        svm = SVM()
        program_1 = knn.knn_predict(k)
        program_2 = svm.svm_predict(g)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(program_1)
            executor.submit(program_2)
            pid = os.getpid()
            print_stats(pid, start)
            # print(pid)


