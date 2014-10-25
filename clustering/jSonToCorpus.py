import json
#jSonFile must be parsed in a raw format or with double backslash
def jsonToCorpus(outputFolder,jSonFile):
    fileName='tweet'
    number=0
    text=open(jSonFile)
    text=text.read()
    text=json.loads(text)
    for i in text:
        tweet=text[i]['text']
        number=number+1
        f = open(outputFolder+'\\'+fileName+str(number),'w')
        f.write(str(tweet.encode('ascii','ignore'))) # python will convert \n to os.linesep
        f.close() # you can omit in most cases as the destructor will call if