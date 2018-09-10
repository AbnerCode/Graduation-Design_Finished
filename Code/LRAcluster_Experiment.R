

ptm <- proc.time()

dm = read.table("/Users/wangzili/Desktop/毕设/LRAcluster_2/R/test_dataset/test_dataset_DM.csv")
ge = read.table("/Users/wangzili/Desktop/毕设/LRAcluster_2/R/test_dataset/test_dataset_GE.csv")
#me = read.table("COAD_SNF_Methylation_Z_1.csv")
#cnv = read.table("D:/Study/2016/Experiments/Integration/Data/LUAD/After preprocessing/SNF/overlap/LUAD_SNF_CNV_Z_1.csv")

dm = as.matrix(dm)
ge = as.matrix(ge)
#me = as.matrix(me)
#cnv = as.matrix(cnv)

#data = list(m, mi         )
#data = list(m,     me     )
#data = list(m,         cnv)
#data = list(   mi, me     )
#data = list(   mi,     cnv)
#data = list(       me, cnv)

#data = list(m, mi, me     )
#data = list(m, mi,     cnv)
#data = list(m,     me, cnv)
#data = list(   mi, me, cnv)

#data = list(m, mi, me)
data = list(dm, ge)
#types = list("gaussian", "gaussian")
#types = list("gaussian", "gaussian", "gaussian")
types = list("gaussian", "gaussian")   #意义是什么。是否能决定类型 数据的类型么

#names = list("mRNAExpression", "miRNAExpression"                         )
#names = list("mRNAExpression",                    "DNAMethylation"       )
#names = list("mRNAExpression",                                      "CNV")
#names = list(                  "miRNAExpression", "DNAMethylation"       )
#names = list(                  "miRNAExpression",                   "CNV")
#names = list(                                     "DNAMethylation", "CNV")
#names = list("mRNAExpression", "miRNAExpression", "DNAMethylation"       )
#names = list("mRNAExpression", "miRNAExpression",                   "CNV")
#names = list("mRNAExpression",                    "DNAMethylation", "CNV")
#names = list(                  "miRNAExpression", "DNAMethylation", "CNV")
names = list("dm", "ge")

res = LRAcluster(data,types, dimension=5, names=as.character(1:length(data)))  
#dimension代表维度 代表将初始的矩阵降维道几维
#得到降维矩阵后进行k-means聚类


result = t(res$coordinate)
kc <- kmeans(result, 8)    #为什么是8类？
time = proc.time() - ptm
#write.csv(t(result), "D:\\Study\\2016\\Experiments\\Integration\\Results\\LUAD\\LRAcluster\\Matrix_m_mi_me_cnv.csv", quote = FALSE)
write.csv(kc$cluster, "8_mi_me_cnv.csv", quote = FALSE)   #k-means结果类型
# quote:数据在写入文件中时我们常用引号将其隔开，当参数为F时，文件中的数据不再用引号修饰

plot(result, col = kc$cluster)
points(kc$centers, col=3:4, pch=8, cex=2)      
#cex代表点的大小 pch代表点的形状 col设置边框的颜色
print(time)
print(res$potential)