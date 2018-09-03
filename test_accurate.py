#coding=utf-8
import os
from lan_id import LanguageIdentifier
#语言对照表
TESTING_LANGUAGES = {"it": "Italian", "pl": "Polish", "ru": "Russian", "sk": "Slovak", "pt": "Portuguese",
                    "ro": "Romanian", "da": "Danish", "sv": "Swedish", "no": "Norwegian", "en": "English",
                    "es": "Spanish", "fr": "French", "cs": "Czech", "de": "German", "fi": "Finnish", "et": "Estonian",
                    "lv": "Latvian", "lt": "Lithuanian", "fa": "Persian", "hu": "Hungarian", "he": "Hebrew",
                    "el": "Greek", "ar": "Arabic"}

def test_data():
    dir_path = 'test/'
    right = 0
    wrong = 0
    print "Test in %s" % dir_path
    print "Language, total, precision"
    global identifier
    # 看是否是终端输入
    if os.path.exists("train/data.model/model"):
        identifier = LanguageIdentifier.from_modelpath("train/data.model/model", norm_probs=True)
        print("使用训练的model")
    else:
        print("未检测到模型")
    for language in TESTING_LANGUAGES:
        print "Language:", language
        lang_right = 0
        lang_wrong = 0
        path = dir_path + language + "/"
        for filename in os.listdir(path):
            with open(path + filename, 'r') as f:
                text = f.read()[0:]
                f.close()
                try:
                    detected_language = identifier.classify(text)[0]
                    if detected_language == language: lang_right += 1
                    else: lang_wrong += 1

                except ZeroDivisionError:
                    print "ERROR: %s" % path + filename

        right += lang_right
        wrong += lang_wrong

        print "Right: %s, Wrong: %s Total: %s" % (str(lang_right), str(lang_wrong), str(lang_right + lang_wrong))
        print "Accuracy: %s " % str(lang_right / float(lang_right + lang_wrong) * 100)
        # print "%s, %s, %.2f" % (TESTING_LANGUAGES[language], str(lang_right + lang_wrong), lang_right / float(lang_right + lang_wrong) * 100)


    print "======TOTAL:"
    print "\tRight: %s, Wrong: %s Total: %s" % (str(right), str(wrong), str(right + wrong))
    print "\tPrecision: %s \n" % str(right / float(right + wrong) * 100)

if __name__ == "__main__":
    test_data()