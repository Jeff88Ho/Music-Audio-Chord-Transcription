__author__ = 'Jeff'


from scipy.io import loadmat
import os

dir = "/Users/Jeff/Desktop/svm_hmm/CHORDS"

num = 0

for file in os.listdir(dir):
    num+=1

    file_name = os.path.join(file)
    print file_name


    read_path = os.path.join(dir, file)
    print read_path


    mat = loadmat(read_path)


    F = mat['F']
    L = mat['L']



    F = F.transpose()



    print F.shape
    print L.shape

    L = L[range(len(L)),0]


    write_path = "/Users/Jeff/Desktop/svm_hmm/input_6/" + str(file_name) + ".txt"

    text_file = open(write_path, "w")


    for i in range(int(F.shape[0])):
        text_file.write(str(L[i]+1)+" "+"qid:"+str(num)) #all label values plus 1

        T = []
        for j in range(int(F.shape[1])):
            text_file.write(" "+str(j+1)+":"+str(F[i, j]))

        for j in range(int(F.shape[1])):
            if i+1 in range(int(F.shape[0])):
                text_file.write(" "+str(j+1+int(F.shape[1]))+":"+str(F[i+1, j]))

        if i+1 in range(int(F.shape[0])):
            double_vec = list(F[i]) + list(F[i+1])
        else:
            double_vec = list(F[i])

        for j in range(len(double_vec)):
            for k in range(j+1):
                T.append(double_vec[j]*double_vec[k])

        for t in range(len(T)):
            text_file.write((" "+str(t+25))+":"+str(T[t]))

        text_file.write(str("\n"))



    text_file.close()
