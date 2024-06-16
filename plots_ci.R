
# install.packages('reticulate')

library(reticulate)
np <- import("numpy")
# data reading

path_amnesiac <- '/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/results/PTB_XL/amnesiac_unlearning/save_accuracy.npy'
path_amnesiac_mia <- '/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/results/PTB_XL/amnesiac_unlearning/save_mia.npy'

path_dp <- '/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/results/PTB_XL/data_pruning/save_accuracy.npy'
path_dp_mia <- '/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/results/PTB_XL/data_pruning/save_mia.npy'



mat_amnesiac <- np$load(path_amnesiac)

# Amnesiac accuracy
round(apply(mat_amnesiac, 2, mean), digits=4)

n <- dim(mat_amnesiac)[1]

round(qt(0.975, n-1) * apply(mat_amnesiac, 2, sd) * 1 / sqrt(n), digits=4)

id <- 1
qqnorm(mat_amnesiac[,id])
qqline(mat_amnesiac[,id])

hist(mat_amnesiac[,id], prob=TRUE)
m<-mean(mat_amnesiac[,id])
std<-sqrt(var(mat_amnesiac[,id]))

curve(dnorm(x, mean=m, sd=std), add=TRUE)






mat_dp <- np$load(path_dp)

# DP accuracy
round(apply(mat_dp, 2, mean), digits=4)

n <- dim(mat_dp)[1]

round(qt(0.975, n-1) * apply(mat_dp, 2, sd) * 1 / sqrt(n), digits=4)



# Amnesiac MIA accuracy
mat_amnesiac_mia <- np$load(path_amnesiac_mia)

round(apply(mat_amnesiac_mia, 2, mean), digits=4)

n <- dim(mat_amnesiac)[1]

round(qt(0.975, n-1) * apply(mat_amnesiac_mia, 2, sd) * 1 / sqrt(n), digits=4)



# dp MIA accuracy
mat_dp_mia <- np$load(path_dp_mia)

round(apply(mat_dp_mia, 2, mean), digits=4)

n <- dim(mat_amnesiac)[1]

round(qt(0.975, n-1) * apply(mat_amnesiac_mia, 2, sd) * 1 / sqrt(n), digits=4)





# Classifier  accuracy on different datsets

ptb_acc <- np$load('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/results/PTB_XL/ts_test_acc_new.npy')

n <- dim(ptb_acc)

round(mean(ptb_acc), digits = 4)

round(qt(0.975, n-1) * sd(ptb_acc) * 1 / sqrt(n), digits=4)



crop_acc <- np$load('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/results/Crop/ts_test_acc_new.npy')

n <- dim(crop_acc)

round(mean(crop_acc), digits = 4)

round(qt(0.975, n-1) * sd(crop_acc) * 1 / sqrt(n), digits=4)



ed_acc <- np$load('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/results/ElectricDevices/ts_test_acc_new.npy')

n <- dim(ed_acc)

round(mean(ed_acc), digits = 4)

round(qt(0.975, n-1) * sd(ed_acc) * 1 / sqrt(n), digits=4)


# Init classifier  accuracy on different datsets

ptb_acc <- np$load('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/results/PTB_XL/init_save.npy')

n <- dim(ptb_acc)

round(mean(ptb_acc), digits = 4)

round(qt(0.975, n-1) * sd(ptb_acc) * 1 / sqrt(n), digits=4)



crop_acc <- np$load('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/results/Crop/init_save.npy')

n <- dim(crop_acc)

round(mean(crop_acc), digits = 4)

round(qt(0.975, n-1) * sd(crop_acc) * 1 / sqrt(n), digits=4)



ed_acc <- np$load('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/results/ElectricDevices/init_save.npy')

n <- dim(ed_acc)

round(mean(ed_acc), digits = 4)

round(qt(0.975, n-1) * sd(ed_acc) * 1 / sqrt(n), digits=4)
