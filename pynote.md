##### sum 和 np.sum 的不同

1. sum 不能处理二维数组；np.sum 可以处理，且默认按列求和。
2. 二者处理矩阵的效果是不同的 a = [[1,2,3],[4,5,6]] m = np.mat([[1,2,3],[4,5,6]]) sum(m) # matrix([[5, 7, 9]])，不能选择axis np.sum(m) # 21，可以选择axis，返回一个array sum()函数和np.sum()的逻辑还是很混乱的：sum 可以处理矩阵但不能处理二维数组，其在处理一维数组时返回整体的和，但在处理矩阵时返回每一列的和； np.sum 在处理数组时返回每一行（或列）的和，但在处理矩阵时却默认情况返回整体的和

##### sort

python D.sort(), D.argsort(), M = D.sort(), M = D.argsort() 均会将排序后的值赋值给D。但是 numpy.sort(D), numpy.argsort(D) 不会改变D的值，这是因为实例D自身的方法通常会改变self的值，我记得有一些方法不会，但我忘记了。

##### sparse matrix

matlab 可以用 sparse 和 full 两个函数来实现普通矩阵和稀疏矩阵之间的转换。python 中稀疏矩阵的实现可以借助 scipy.sparse，其有多种不同的存储形式，其中与 matlab 中的相对应的是 coo_matrix，用坐标来表示矩阵。得到一个矩阵 SM，SM.todense() 即可得到 full matrix。

coo_matrix(M) 即可得到一个矩阵 M 的稀疏矩阵。

##### np.max() 和 np.maximun()

np.max 求序列的最值，相当于 matlab 中的 max(A)

np.maximum(X, Y) X 与 Y 逐位比较取其大者，相当于 matlab 中的 max(X, Y)

##### np.where

np.where(condition) 返回满足条件的数组<u>下标</u> （或矩阵坐标，以两个向量的形式返回）

np.where(condition, a, b):

​	a = np.array([1,3,2]), b = np.where(a > 1, a, 9) = array([9, 3, 2])

​	相当于 if condition: do a;    else: do b;

np.where(condition, )

##### np.tile()

np.tile(a, (3,1)) 相当于 matlab 中的 repmat(a, 3, 1)

##### pycharm

pycharm 中的代码文件不要以 test 开头，因为 pycharm 会把所有以 test 开头的文件默认加入 test 测试模块，无法直接运行。