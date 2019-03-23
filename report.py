import numpy as np
from pytz import timezone
from pandas import isnull
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='OmniFocus.csv', help="The name to the csv file")
parser.add_argument("--timezone", type=str, default='US/Pacific')
parser.add_argument("--target", type=float, default=4, help="Target work hours per day")
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
            if not self.completion_date:
                self.completed_time = 0
                days_away = (self.due_date.date() - datetime.now(self.timezone).date()).days
                self.earliest_due = days_away
                # print(days_away, self.due_date.date(), datetime.now(self.timezone).date())
                return None, np.array([[days_away, self.total_time]], dtype=np.int)
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

    def plot_completion(self):
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        xaxis = [-i for i in range(len(self.completion_pdf))]
        ax2.bar(xaxis, np.minimum(self.completion_pdf, 24), alpha=0.5, color='m')
        ax2.set_ylabel('Hours by Day', fontsize=14)

        ax1.plot(xaxis, self.completion_cdf, linewidth=3)
        ax1.set_xlabel('Days ago', fontsize=14)
        ax1.set_ylabel('Cumulative Hours', fontsize=14)

        if self.level == 0:
            if len(xaxis) > 30:
                ax1.plot(xaxis[:30], self.completion_cdf[30] + np.flip(np.array([i * args.target for i in range(30)]), 0),
                         color='g', linewidth=3, linestyle=':', alpha=0.8)
            if len(xaxis) > 7:
                ax1.plot(xaxis[:7], self.completion_cdf[7] + np.flip(np.array([i * args.target for i in range(7)]), 0),
                         color='g', linewidth=3, linestyle=':', alpha=1.0)
            ax1.plot(xaxis, np.flip(np.array([i * args.target for i in range(len(self.completion_pdf))]), 0),
                     color='g', alpha=0.6, linewidth=3, linestyle=':', label='Reference %.1fh/day' % args.target)
        ax2.axhline(8.0, linestyle=':', color='m')

        if self.level == 0:
            ax1.legend()

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
                         color='g', linewidth=3, linestyle=':', alpha=1.0)

        ax2.axhline(8.0, linestyle=':', c='m')

    def generate_report(self, depth=0, parent_name=""):
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.title("Completed Time")
        try:
            self.plot_completion()
        except:
            plt.cla()
        plt.subplot(1, 3, 2)
        plt.title("Due Work (Week)")
        try:
            self.plot_due(7)
        except:
            plt.cla()
        plt.subplot(1, 3, 3)
        plt.title("Due Work (Month)")
        try:
            self.plot_due(30)
        except:
            plt.cla()
        plt.tight_layout()

        if not os.path.isdir(parent_name):
            os.makedirs(parent_name)
        name = parent_name + "/" + self.name
        plt.savefig('%s' % name + ".png")

        if depth > 0:
            for index in self.children.keys():
                self.children[index].generate_report(depth-1, name)
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


