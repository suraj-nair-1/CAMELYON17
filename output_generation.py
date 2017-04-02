import numpy as np

preds = np.loadtxt("predictions/run5raw.csv", dtype="int", delimiter=",")

node_dc = {}
pat_dc = {}

labels = {3:"macro", 2:"micro", 1:"itc", 0:"negative"}

submissionnum = 2

f = open('predictions/submission'+str(submissionnum)+'.csv', 'a')
f.write("patient, stage\n")

for row in preds:
    patient = row[0]
    node = row[1]
    pred = row[2]
    ky = str(patient) + "_" + str(node)
    if node_dc.has_key(ky):
        node_dc[ky] += [pred]
    else:
        node_dc[ky] = [pred]

node_dc["100_4"] = 0
node_dc["140_4"] = 0
node_dc["183_0"] = 0

print node_dc

for key in node_dc:
    vals = node_dc[key]
    mx = np.max(vals)
    pat_ky, nod_ky = key.split("_")
    # f = open('predictions/submission'+str(submissionnum)+'.csv', 'a')
    # f.write("patient_" + pat_ky + "_node_" + nod_ky + ".tif," + labels[mx] + "\n")
            
    if pat_dc.has_key(pat_ky):
        pat_dc[pat_ky] += [mx]
    else:
        pat_dc[pat_ky] = [mx]

print pat_dc

for key in pat_dc.keys():
    vals = pat_dc[key]
    if len(vals) < 5:
        print(key)
    mx = np.max(vals)
    count3 = 0
    for v in vals:
        if v == 3:
            count3 += 1

    if mx == 0:
        label = "pN0"
    elif mx == 1:
        label = "pN0(i+)"
    elif mx == 2:
        label = "pN1mi"
    elif count3 < 3:
        label = "pN1"
    else:
        label = "pN2"
    # f = open('predictions/submission'+str(submissionnum)+'.csv', 'a')
    # f.write("patient_" + key + ".zip," + label + "\n")



