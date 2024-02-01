from SELFRec import SELFRec
from util.conf import ModelConf

if __name__ == '__main__':
    # 改写为动态传参的形式
    model = 'SGCL'
    import time
    conf = ModelConf('./conf/' + model + '.conf')
    print(conf)
    s = time.time()
    rec = SELFRec(conf)
    print(rec)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
