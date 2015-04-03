import os,re

def convertFoamToCSV(filenameNameIn, filenameOut):
    fd = open(filenameNameIn, 'r')
    numL=0
    storeLine=[]
    singleton=True
    for line in fd:
       numL = numL +1
       if ((numL<16) or (len(line)<=1) or re.match("(.*)//",line)):
          continue
       else:
          storeLine.append(line) 
          if (re.match("(.*){(.*)}(.*)",line) and singleton): #if it's a singleton
             print "It's a singleton"
             splitLine = re.split("{|}|    |\n", line)
             nbEle = int(splitLine[0])
             file_name_out = filenameOut#str(re.split("/",filenameNameIn)[1]) + ".csv" #remplace with 'filename'
             if len(splitLine)>4: #a vectorField
                valueX = float(splitLine[2])
                valueY = float(splitLine[3])
                valueZ = float(splitLine[4])
                outputFile = open(file_name_out, 'w')
                for i in xrange(0,nbEle,1):
                   outputFile.write(str(valueX) + "," + str(valueY) + "," + str(valueZ) +"\n")
                outputFile.close()
                #break
             else: #a scalarField
                value = float(splitLine[1])
                outputFile = open(file_name_out, 'w')
                for i in xrange(0,nbEle,1):
                   outputFile.write(str(value)+"\n")
                outputFile.close()
                #break
          else: #it's not a singleton
             if (singleton):
                print "it's not a singleton"
                singleton = False
                #break
    if (not singleton):
       n = numL
       var = n-23
       en = n-4
       str1 = "head -n" + str(en) + " " + str(filenameNameIn) + " > " + str(filenameOut) + ".1"
       str2 = "tail -n" + str(var) + " " + str(filenameOut) + ".1 > " + str(filenameOut) + ".F"
       str3 = "rm -f " + str(filenameOut) + ".1"
       str4 = "sed -e 's/(//g' -e 's/)//g' " + str(filenameOut) + ".F > " + str(filenameOut)
       str5 = "rm -f " + str(filenameOut) + ".F"
       os.system(str1)
       os.system(str2)
       os.system(str3)
       os.system(str4)
       os.system(str5)
       fd.close()

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = newPath

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
