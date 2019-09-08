import numpy as np
from pytz import timezone
from pandas import isnull
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='OmniFocus.csv', help="The name to the csv file")
parser.add_argument("--timezone", type=str, default='US/Pacific')
parser.add_argument("--target", type=float, default=7, help="Target work hours per day")
args = parser.parse_args()
print(args)

class TreeNode:
    def __init__(self, level):
        self.level = level
        self.children = {}

        # Derivative quantities
        self.completed_time = 0  # Total amount of work completed
        self.completion_cdf = []  # Amount of work completed until i days ago
        self.completion_pdf = []
        self.scheduled_time = 0  # Total time that has been scheduled as leaf tasks
        self.due_cdf = []  # Amount of work to complete before i days away
        self.due_pdf = []
        self.earliest_due = 0

        # Original quantities
        self.total_time = 0
        self.name = "projects"
        self.task_id = ""
        self.due_date = None
        self.completion_date = None

        self.timezone = timezone(args.timezone)

    def add_children(self, data, id_list):
        if len(id_list) == 1:
            cur_id = id_list[0]
            if cur_id in self.children:
                print(id_list, data)
                assert False
            self.children[cur_id] = TreeNode(self.level + 1)
            self.children[cur_id].fill_info(data)
        else:
            cur_id = id_list[0]
            if cur_id not in self.children:
                print(id_list, data)
                assert False
            self.children[cur_id].add_children(data, id_list[1:])

    def parse_time(self, text_time):
        return datetime.strptime(text_time, '%Y-%m-%d %H:%M:%S %z').astimezone(self.timezone)

    def fill_info(self, data):
        self.name = data['Name']
        self.task_id = data['Task ID']
        if not isnull(data['Completion Date']):
            self.completion_date = self.parse_time(data['Completion Date'])
        if not isnull(data['Duration']):
            self.total_time = int(data['Duration'][:-1])
        if not isnull(data['Due Date']):
            self.due_date = self.parse_time(data['Due Date'])

    def clean_tree(self):
        # This should only be called by root note
        # Complete any missing due dates
        self.inherit_due(
            datetime.strptime("2019-12-31 11:59:00 +0000", '%Y-%m-%d %H:%M:%S %z').astimezone(self.timezone))

    def inherit_due(self, due_date):
        if self.due_date is None:
            self.due_date = due_date
        for index in self.children.keys():
            self.children[index].inherit_due(self.due_date)

    # def compute_scheduled_time(self, cur_total):
    #     if len(self.children) == 0:
    #         self.scheduled_time = self.total_time
    #     else:
    #         self.scheduled_time
    #         for index in self.children.keys():
    #             completion_list = self.children[index].compute_completion()
    #             if completion_list is not None:
    #                 cur_list.append(completion_list)
    #         if len(cur_list) == 0:
    #             return None
    #         cur_list = np.concatenate(cur_list, axis=0)
    #         self.completed_time = np.sum(cur_list[:, 1])
    #         cur_list = cur_list[np.argsort(cur_list[:, 0])]
    #
    #         # Compute CDF
    #         self.completion_pdf = np.bincount(cur_list[:, 0], weights=cur_list[:, 1]) / 60.0
    #         self.completion_cdf = np.cumsum(self.completion_pdf[::-1])[::-1]
    #         return cur_list

    def compute_completion(self):
        # This is a leaf node
        if len(self.children) == 0:
            # If the action has not been completed, add it to future tasks times
            if not self.completion_date:
                self.completed_time = 0
                days_away = (self.due_date.date() - datetime.now(self.timezone).date()).days
                self.earliest_due = days_away
                # print(days_away, self.due_date.date(), datetime.now(self.timezone).date())
                return None, np.array([[days_away, self.total_time]], dtype=np.int)
            # If the action is completed, add it to past tasks times
            else:
                self.completed_time = self.total_time
                self.earliest_due = 0
                days_ago = (datetime.now(self.timezone).date() - self.completion_date.date()).days
                return np.array([[days_ago, self.total_time]], dtype=np.int), None
        else:
            completion_list = []
            due_list = []
            for index in self.children.keys():
                completion, due = self.children[index].compute_completion()
                if completion is not None:
                    completion_list.append(completion)
                if due is not None:
                    due_list.append(due)
            if len(completion_list) != 0:
                completion_list = np.concatenate(completion_list, axis=0)
                self.completed_time = np.sum(completion_list[:, 1])
                completion_list = completion_list[np.argsort(completion_list[:, 0])]

                # Compute CDF
                self.completion_pdf = np.bincount(completion_list[:, 0], weights=completion_list[:, 1]) / 60.0
                self.completion_cdf = np.cumsum(self.completion_pdf[::-1])[::-1]
            else:
                completion_list = None

            if len(due_list) != 0:
                due_list = np.concatenate(due_list, axis=0)
                self.earliest_due = np.min(due_list[:, 0])
                self.due_pdf = np.bincount(due_list[:, 0] - self.earliest_due, weights=due_list[:, 1]) / 60.0
                self.due_cdf = np.cumsum(self.due_pdf)
            else:
                due_list = None
            return completion_list, due_list

    # Report generation
    def plot_completion(self, ylim1=None, date_range=None):
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        if date_range is None:
            pdf = self.completion_pdf
            cdf = self.completion_cdf
            date_range = len(self.completion_cdf)
        else:
            date_range = min(date_range, len(self.completion_pdf))
            pdf = self.completion_pdf[:date_range]
            cdf = self.completion_cdf[:date_range].copy()
            cdf -= self.completion_cdf[date_range]
        xaxis = [-i for i in range(len(pdf))]
        ax2.bar(xaxis, np.minimum(pdf, 24), alpha=0.5, color='m')
        ax2.set_ylabel('Hours by Day', fontsize=14)

        ax1.plot(xaxis, cdf, linewidth=3)
        ax1.set_xlabel('Days ago', fontsize=14)
        ax1.set_ylabel('Cumulative Hours', fontsize=14)

        if self.level == 0:
            if len(xaxis) > 30:
                ax1.plot(xaxis[:30], cdf[30] + np.flip(np.array([i * args.target for i in range(30)]), 0),
                         color='g', linewidth=3, linestyle=':', alpha=0.8)
            if len(xaxis) > 7:
                ax1.plot(xaxis[:7], cdf[7] + np.flip(np.array([i * args.target for i in range(7)]), 0),
                         color='g', linewidth=3, linestyle=':', alpha=1.0)
            ax1.plot(xaxis, np.flip(np.array([i * args.target for i in range(len(pdf))]), 0),
                     color='g', alpha=0.6, linewidth=3, linestyle=':')
            ax2.axhline(args.target, linestyle=':', color='g', label='Reference %.1fh/day' % args.target)
            ax2.axhline(cdf[0] / date_range, linestyle=':', color='m', label='Actual %.1fh/day' % (cdf[0] / date_range))
        if ylim1 is not None:
            ax1.set_ylim([0, ylim1])
        if self.level == 0:
            ax2.legend()

    def plot_category(self, time_range=None, min_fraction=5.0):
        items = {}
        total_time = 0
        for index in self.children:
            cdf = self.children[index].completion_cdf
            if len(cdf) != 0:
                if time_range is not None and len(cdf) > time_range:
                    sub = cdf[time_range]
                else:
                    sub = 0.0
                items[self.children[index].name] = cdf[0] - sub
                total_time += cdf[0] - sub

        # Assign a unique color to each item
        palette = sns.color_palette("husl", len(items))
        color_dict = {}
        for i, key in enumerate(items):
            color_dict[key] = palette[i]

        threshold = total_time * min_fraction / 100.0
        labels = ['Others']
        values = [0.0]
        colors = ['gray']
        for key in items:
            # If text too long line break it
            if len(key) < 10 or key.find(' ') == -1:
                name = key
            else:
                prev_find = -1
                while True:
                    cur_find = key.find(' ', prev_find + 1)
                    if cur_find == -1:
                        break
                    prev_find = cur_find
                    if cur_find > len(key) / 2.0:
                        break
                name = key[:prev_find] + '\n' + key[prev_find + 1:]

            if items[key] > threshold:
                labels.append(name)
                values.append(items[key])
                colors.append(color_dict[key])
            else:
                values[0] += items[key]

        if values[0] < 0.1:
            labels = labels[1:]
            values = values[1:]
            colors = colors[1:]

        ax1 = plt.gca()

        def func(pct):
            absolute = int(pct / 100. * total_time)
            return "{:.0f}%\n({:d} h)".format(pct, absolute)

        ax1.pie(values, labels=labels, autopct=lambda pct: func(pct),
                shadow=False, startangle=90, colors=colors, pctdistance=0.85)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    def plot_due(self, date_range=7):
        if date_range > self.earliest_due + len(self.due_pdf):
            date_range = self.earliest_due + len(self.due_pdf)
        drange = range(self.earliest_due, date_range + 1)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.bar(drange, np.minimum(self.due_pdf[:len(drange)], 24), alpha=0.5, color='m')
        ax2.set_ylabel('Hours by Day', fontsize=14)
        ax2.set_ylim([0, 15])

        ax1.plot(drange, self.due_cdf[:len(drange)], linewidth=3)
        ax1.set_xlabel('Days in the future', fontsize=14)
        ax1.set_ylabel('Cumulative Hours', fontsize=14)
        ax1.set_ylim([0, np.max(self.due_cdf[:len(drange)]) * 1.2])

        if self.level == 0:
            ax1.plot(range(date_range), [i * args.target for i in range(1, date_range+1)],
                         color='g', linewidth=3, linestyle=':', alpha=1.0, explode=True)

            ax2.axhline(args.target, linestyle=':', c='m')

    def generate_report(self, depth=0, parent_name="Reports"):
        from matplotlib import rcParams
        rcParams['axes.titlepad'] = 15

        plt.figure(figsize=(21, 12))
        plt.subplot(2, 3, 1)
        plt.title("Completed Time (Total)")
        try:
            self.plot_completion()
        except:
            plt.cla()

        plt.subplot(2, 3, 2)
        plt.title("Completed Time (Past 30 Days)")
        try:
            self.plot_completion(date_range=30)
        except:
            plt.cla()

        plt.subplot(2, 3, 3)
        plt.title("Due Work (Week)")
        try:
            self.plot_due(7)
        except:
            plt.cla()

        plt.subplot(2, 3, 4)
        plt.title("Time by Category (Total)")
        try:
            self.plot_category(min_fraction=3.0)
        except:
            plt.cla()

        plt.subplot(2, 3, 5)
        plt.title("Time by Category (Past 30 Days)")
        try:
            self.plot_category(time_range=30, min_fraction=3.0)
        except:
            plt.cla()

        plt.subplot(2, 3, 6)
        plt.title("Time by Category (Past 7 Days)")
        try:
            self.plot_category(time_range=7)
        except:
            plt.cla()
        # plt.subplot(1, 3, 3)
        # plt.title("Due Work (Month)")
        # try:
        #     self.plot_due(30)
        # except:
        #     plt.cla()
        plt.tight_layout(pad=5.0, w_pad=5.0)

        if not os.path.isdir(parent_name):
            os.makedirs(parent_name)
        name = parent_name + "/" + self.name
        plt.savefig('%s' % name + ".png")

        if depth > 0:
            for index in self.children.keys():
                self.children[index].generate_report(depth-1, name)
        plt.close()

    def generate_itemized(self, depth=0, parent_name="Reports"):
        plt.figure(figsize=(40, 20))
        num_reports = len(self.children)
        height = int(math.floor(math.sqrt(num_reports)))
        width = int(math.ceil(num_reports / float(height)))

        max_time = 0
        for i, index in enumerate(self.children.keys()):
            if len(self.children[index].completion_cdf) != 0 and self.children[index].completion_cdf[0] > max_time:
                max_time = self.children[index].completion_cdf[0]
        max_time *= 1.1

        for i, index in enumerate(self.children.keys()):
            plt.subplot(height, width, i+1)
            plt.title(self.children[index].name, fontsize=20, fontstyle='oblique')
            self.children[index].plot_completion(ylim1=max_time)

        plt.tight_layout()

        if not os.path.isdir(parent_name):
            os.makedirs(parent_name)
        name = parent_name + "/" + self.name
        plt.savefig('%s' % name + "_itemized.png")

        if depth > 0:
            for index in self.children.keys():
                self.children[index].generate_itemized(depth-1, name)
        plt.close()

    # Printing functions for debugging
    def serialize_time(self, time):
        if time > 60:
            return "%dh%dm" % (time // 60, time % 60)
        else:
            return "%dm" % time

    def serialize_info(self):
        msg = "Id %10s | Name %30s | Duration %6s | Due date %20s | Completed date  %20s | completion %6s | Children %d" % \
              (self.task_id, self.name[:30], self.serialize_time(self.total_time), str(self.due_date),
               str(self.completion_date), self.serialize_time(self.completed_time), len(self.children))
        return msg

    def print_tree(self):
        print(self.serialize_info())
        for index in self.children.keys():
            self.children[index].print_tree()


result = pd.read_csv(args.path, delimiter = ',')
task_tree = TreeNode(0)

for i in range(result.shape[0]):
    if isnull(result.iloc[i]['Task ID']):
        continue
    item_id = [int(item) for item in result.iloc[i]['Task ID'].split('.')]
    task_tree.add_children(result.iloc[i], item_id)
task_tree.clean_tree()
task_tree.compute_completion()

task_tree.generate_report(depth=1, parent_name='Reports')
task_tree.generate_itemized(depth=0, parent_name='Reports')

