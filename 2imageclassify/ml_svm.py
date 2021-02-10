# 课程可以围绕这个文档展开
# https://scikit-learn.org/stable/modules/svm.html
# 内部计算是依赖libsvm和liblinear库的
# libsvm:https://www.csie.ntu.edu.tw/~cjlin/libsvm/
# liblinear:https://www.csie.ntu.edu.tw/~cjlin/liblinear/
from sklearn import svm
X = [[0,0],[1,1]]
y = [0,1]
# 二分类
clf = svm.SVC()
clf.fit(X,y)
clf.predict([[2.,2.]])

# 获取支持向量
clf.support_vectors_
# 获取支持向量的序号
clf.support_
# 获取每一类的支持向量
clf.n_support_


# 多分类
# https://scikit-learn.org/stable/modules/svm.html
X = [[0],[1],[2],[3]]
Y = [0,1,2,3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X,Y)

dec = clf.decision_function([[1]])
dec.shape[1]
