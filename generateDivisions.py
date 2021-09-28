import argparse, os, math, shutil, random
from function import getDirectoriesInPath, getFilesInPath

def affwild2(pathbase,typeExp,outputfolder):
    pathFilesForTraining = os.path.join(pathbase,'cropped_aligned')
    if os.path.exists(outputfolder):
        shutil.rmtree(outputfolder)

    os.makedirs(os.path.join(outputfolder,'train'))
    os.makedirs(os.path.join(outputfolder,'val'))
    os.makedirs(os.path.join(outputfolder,'test'))

    setsFile = ['Train_Set','Validation_Set']
    typesFolder = {'EXP' : 'EXPR_Set','VA' : 'VA_Set', 'AU' : 'AU_Set'}
    for s in setsFile:
        fullPathFiles = os.path.join(pathbase,'annotations',typesFolder[typeExp],s)
        filesToSeparate = getFilesInPath(fullPathFiles)
        sizeVal = 0
        if 'Train' in s:
            sizeVal = math.floor(len(filesToSeparate) * 0.1)

        for f in filesToSeparate:
            folderName = f.split(os.path.sep)[-1][:-4]
            print("Copying files from %s to dataset" % (folderName))
            filesFromVideo = getFilesInPath(os.path.join(pathFilesForTraining,folderName))
            if sizeVal > 0 and random.randint(0,1):
                for fImage in filesFromVideo:
                    if 'jpg' not in fImage:
                        continue
                    fileJpgName = fImage.split(os.path.sep)[-1]
                    shutil.copy(fImage,os.path.join(outputfolder,'val',folderName+'_'+fileJpgName))
                sizeVal -= 1            
            elif 'Train' in s:
                for fImage in filesFromVideo:
                    if 'jpg' not in fImage:
                        continue
                    fileJpgName = fImage.split(os.path.sep)[-1]
                    shutil.copy(fImage,os.path.join(outputfolder,'train',folderName+'_'+fileJpgName))
            else:
                for fImage in filesFromVideo:
                    if 'jpg' not in fImage:
                        continue
                    fileJpgName = fImage.split(os.path.sep)[-1]
                    shutil.copy(fImage,os.path.join(outputfolder,'test',folderName+'_'+fileJpgName))


def main():
    parser = argparse.ArgumentParser(description='Generate data for training/testing phase')
    parser.add_argument('--pathBase', help='Database path', required=True)
    parser.add_argument('--type', help='Type of evaluation (EXP,VA,AU)', required=True)
    parser.add_argument('--outputFolder', help='Folder to output files', required=True)
    parser.add_argument('--dataset', help='Dataset to process', required=True)    
    args = parser.parse_args()

    eval("%s(\"%s\",\"%s\",\"%s\")" % (args.dataset,args.pathBase,args.type,args.outputFolder))



if __name__ == '__main__':
    main()