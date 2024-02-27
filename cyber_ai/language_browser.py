import webbrowser

if __name__=="__main__":

    while True:

        print("Please enter next word")
        search = input()

    

        webbrowser.open_new(r"http://translate.google.com/translate?hl=en&sl=zh-TW&tl=en&u=http%3A%2F%2Fwww.google.com.hk%2Fsearch%3Fq%3D" + search + r"%26num%3D10%26hl%3Dzh-TW%26tbo%3Dd%26site%3Dimghp%26tbm%3Disch%26sout%3D1%26biw%3D1075%26bih%3D696")
        webbrowser.open_new_tab(r"https://www.mdbg.net/chinese/dictionary?page=worddict&wdrst=1&wdqb=" + search)
        webbrowser.open_new_tab(r"https://translate.google.com/?sl=zh-TW&tl=en&text=" + search + r"&op=translate")
        webbrowser.open_new_tab(r"http://www.forvo.com/word/" + search + r"/#zh")
