import webbrowser
import pandas as pd
import re

if __name__=="__main__":

    df = pd.read_csv(r"/home/david/Documents/Cyber-AI/data/mandarin_radicals.csv")
    df.drop(df[df['learnt'] == True].index, inplace = True)

    pattern = r'[^()]+'

    for index, row in df.iterrows():

        print("Press enter when ready for the next word")
        input()

        searches = [row["radical"]]

        print("Searching word(s):")
        print("{0}   {1}".format(row["radical"], row["pinyin"]))

        if not pd.isnull(df.loc[index, 'variants']):
            variants = row["variants"].split(",")
            for variant in variants:
                matches = re.findall(pattern, variant)

                if len(matches) == 2:
                    print("{0}   {1}".format(matches[0], matches[1]))
                    searches.append(matches[0])
                else:
                    print("{0}   {1}".format(matches[0], row["pinyin"]))

        webbrowser.open_new(r"http://translate.google.com/translate?hl=en&sl=zh-TW&tl=en&u=http%3A%2F%2Fwww.google.com.hk%2Fsearch%3Fq%3D" + searches[0] + r"%26num%3D10%26hl%3Dzh-TW%26tbo%3Dd%26site%3Dimghp%26tbm%3Disch%26sout%3D1%26biw%3D1075%26bih%3D696")
        webbrowser.open_new_tab(r"https://www.mdbg.net/chinese/dictionary?page=worddict&wdrst=1&wdqb=" + searches[0])
        webbrowser.open_new_tab(r"https://translate.google.com/?sl=zh-TW&tl=en&text=" + searches[0] + r"&op=translate")
        for search in searches:
            webbrowser.open_new_tab(r"http://www.forvo.com/word/" + search + r"/#zh")

        df.loc[index, "learnt"] = True
        df.to_csv(r"/home/david/Documents/Cyber-AI/data/mandarin_radicals.csv", index=False)