__author__ = 'Jeff'


import os
from random import shuffle
import subprocess
import re
import numpy as np




model_loss_values = []
model_accuracy_values = []
model_means = []
model_stds = []
model_quantiles = []


#perform svm_hmm for different models (different feature combinations), processed by data_1 ~ data_16
for i in range(11,16):
    model = i+1
    dir = "/Users/Jeff/Desktop/svm_hmm/input_" + str(model) #data path for different models (total 16)
    print dir


    path_list = []


    #get info of all .txt files
    for file in os.listdir(dir):
        file_name = os.path.join(file)
        read_path = os.path.join(dir, file)
        path_list.append(read_path)

    all_path = path_list[1:181] #gather all path for reading



    loss_stats = []

    for i in range(10): #do it 10 times to get loss stats

        round = i+1


        #random permutation
        shuffle_list = range(180)
        shuffle(shuffle_list)


        train_list = shuffle_list[0:54] #take first 30% as training data
        test_a_list = shuffle_list[171:180] #take last 5% as training data a
        test_b_list = shuffle_list[162:171] #take last 5% as training data b

        train_list.sort()
        test_a_list.sort()
        test_b_list.sort()

        #prepare training data
        train = []
        for item in train_list:
            train.append(all_path[item])

        #preapre testing data a
        test_a = []
        for item in test_a_list:
            test_a.append(all_path[item])

        #preapre testing data b
        test_b = []
        for item in test_b_list:
            test_b.append(all_path[item])


        #write training data
        with open("/Users/Jeff/Desktop/svm_hmm/train.txt", 'w') as outfile:
            for fname in train:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)

        #write testing data a
        with open("/Users/Jeff/Desktop/svm_hmm/test_a.txt", 'w') as outfile:
            for fname in test_a:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)

        #write testing data b
        with open("/Users/Jeff/Desktop/svm_hmm/test_b.txt", 'w') as outfile:
            for fname in test_b:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)


        #now comes cmd

        def subprocess_cmd(command):
            process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
            proc_stdout = process.communicate()[0].strip()
            return proc_stdout



        #all c values to be validated
        c_list = [ pow(2, -6), pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1), pow(2, 0),
                   pow(2, 1), pow(2, 2), pow(2, 3), pow(2, 4), pow(2, 5), pow(2, 6), pow(2, 7), pow(2, 8) ]



        loss_a_list = []
        loss_b_list = []


        #for all c values do validate to get loss
        for c in c_list:

            #learning cmd
            learn_cmd = "/Users/Jeff/Desktop/svm_hmm/svm_hmm_learn -c %s -e 0.1 /Users/Jeff/Desktop/svm_hmm/train.txt /Users/Jeff/Desktop/svm_hmm/model_%s.txt" % (c,c)

            #classifying set a command
            classify_a_cmd = "/Users/Jeff/Desktop/svm_hmm/svm_hmm_classify /Users/Jeff/Desktop/svm_hmm/test_a.txt /Users/Jeff/Desktop/svm_hmm/model_%s.txt /Users/Jeff/Desktop/svm_hmm/tag_a_%s.txt" % (c,c)

            #classifying set b command
            classify_b_cmd = "/Users/Jeff/Desktop/svm_hmm/svm_hmm_classify /Users/Jeff/Desktop/svm_hmm/test_b.txt /Users/Jeff/Desktop/svm_hmm/model_%s.txt /Users/Jeff/Desktop/svm_hmm/tag_b_%s.txt" % (c,c)

            #learning
            print ""
            print "------------------------------------------model %s, round %s, c=%s, learning...-----------------------------------------------------" % (model, round, c)
            learn_result = subprocess_cmd(learn_cmd)

            #classyfying set a
            print ""
            print "------------------------------------------model %s, round %s, c=%s, classifying_a...------------------------------------------------" % (model, round, c)
            print ""
            classify_a_result = subprocess_cmd(classify_a_cmd)
            print classify_a_result

            #classfying set b
            print ""
            print "------------------------------------------model %s, round %s, c=%s, classifying_b...------------------------------------------------" % (model, round, c)
            print ""
            classify_b_result = subprocess_cmd(classify_b_cmd)
            print classify_b_result


            #retrieve loss values from classifying cmd result
            loss_a = re.findall( r"[-+]?\d*\.\d+|\d+", classify_a_result )[10]
            loss_b = re.findall( r"[-+]?\d*\.\d+|\d+", classify_b_result )[10]


            print ""
            print "loss a is ", loss_a
            print "loss b is ", loss_b


            loss_a = float(loss_a)
            loss_b = float(loss_b)


            #store all losses as lists
            loss_a_list.append(loss_a)
            loss_b_list.append(loss_b)


        print ""
        print "##########################################finish model %s, round %s####################################################" % (model, round)
        print ""


        #find optimal loss values of set a and set b
        loss_b = loss_b_list[loss_a_list.index(min(loss_a_list))] #min value in loss_a_list -> optimal C value of set a -> train and classify set b -> loss_b
        loss_a = loss_a_list[loss_b_list.index(min(loss_b_list))] #min value in loss_b_list -> optimal C value of set a -> train and classify set b -> loss_a


        print "summary of model %s, round %s: " % (model, round)
        print "loss_a_list is " + str(loss_a_list)
        print "loss_b_list is " + str(loss_b_list)


        print ""
        print "loss a is ", loss_a
        print "loss b is ", loss_b


        loss_a = float(loss_a)
        loss_b = float(loss_b)

        #weights for set a and set b
        a_num_lines = sum(1 for line in open('/Users/Jeff/Desktop/svm_hmm/test_a.txt'))
        b_num_lines = sum(1 for line in open('/Users/Jeff/Desktop/svm_hmm/test_b.txt'))

        #final loss of the round
        loss = (loss_a*a_num_lines + loss_b*b_num_lines) / (a_num_lines+b_num_lines)

        print "loss is ", loss

        #store all losses
        loss_stats.append(loss)

        print ""
        print "##########################################end of model %s, round %s####################################################" % (model, round)
        print ""



    #summary and stats for each model
    print ""
    print "##########################################finish model %s####################################################" % model
    print ""
    print "summary of model %s:" % model

    print "loss values: " + str(loss_stats)
    loss_stats.sort()
    print "sorted loss values: " + str(loss_stats)

    accuracy_stats = 1 - np.array(loss_stats)
    print "accuracy values: " + str(accuracy_stats)
    accuracy_stats.sort()
    print "sorted accuracy values: " + str(accuracy_stats)

    print "mean is " + str(np.mean(accuracy_stats))
    print "std is " + str(np.std(accuracy_stats))

    quantiles = (accuracy_stats[2], (accuracy_stats[4]+accuracy_stats[5])/2, accuracy_stats[7])
    print "25%, 50%, 75%, quantiles are " + str(quantiles)


    #store for all summary
    model_loss_values.append(loss_stats)
    model_accuracy_values.append(accuracy_stats)
    model_means.append(np.mean(accuracy_stats))
    model_stds.append(np.std(accuracy_stats))
    model_quantiles.append(quantiles)

    print ""
    print "##########################################end of model %s####################################################" % model
    print ""


#summary and stats for the report
print ""
print "all summary:"

print "all loss values for all models: " + str(model_loss_values)

print "all accuracy values for all models: " + str(model_accuracy_values)

print "means for all models are " + str(model_means)
print "stds for all models are " + str(model_stds)

print "25%, 50%, 75%, quantiles for all models are " + str(model_quantiles)


#END
