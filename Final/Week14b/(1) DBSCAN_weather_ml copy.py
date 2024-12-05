############## 코드 작성 
##############


# visulalization
for clust_number in set(labels):
    c = (([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int64(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color = c, s = 20, alpha = 0.85)
    if clust_number != -1 :
        cenx = np.mean(clust_set.xm)
        ceny = np.mean(clust_set.ym)
        plt.text(cenx,ceny, str(clust_number), fontsize=25, color='red')
        plt.show()