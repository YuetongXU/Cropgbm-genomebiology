library("rrBLUP")
library("parallel")
library("pbapply")
## overall parameter
cpus <- 25
## load data
CV <- as.matrix(read.table("maize_CVFs.csv",header = T,sep = ",",row.names = 1))
G <- as.matrix(read.table("maize_geno.csv",header = T,sep = ",",row.names = 1))
phe <- read.table("maize_pheno.csv",header = T,sep = ",",stringsAsFactors = F)

cat("All data loaded!")

## set gradient
phe_list <- c('HT', 'FT', 'YLD')
gradient <- c(10, 50, 100, 500, 1000, 2000)
rd_gradient <- 0:9
## run
## parallel
## write header of file 
write.table(matrix(c("CVid",paste0("CV",1:100)),nrow = 1),file = "G3_maize_CV.csv",row.names = F,col.names = F,quote = F,sep = ",",append = F)
for(phe_name in phe_list){
      G_tmp <- G
      phe_tmp <- phe[,phe_name]
      ## run
      cl <- makeForkCluster(cpus)
      pboptions(type = "timer")
      results <- pbapply(CV,2,function(x){
        library("rrBLUP")
        
        rrBLUP_func <- function (x, y, idx1, idx2){
          .trainModel_RRBLUP <- function( markerMat, phenVec,X = NULL){
            phen_answer<-mixed.solve(phenVec, Z=markerMat, K=NULL, SE = FALSE, return.Hinv=FALSE,X = X)
            beta <- phen_answer$beta
            phD <- phen_answer$u
            e <- as.matrix(phD)
            return( list(beta = beta, e = e, phD = phD) )
          }
          
          trainG <- x[idx1, ]
          testG <- x[idx2, ]
          ytrain <- as.numeric(y[idx1])
          res <- .trainModel_RRBLUP(markerMat = trainG, phenVec = ytrain, 
                                    X = NULL)
          ypred <- testG %*% res$e
          ypred <- ypred[, 1] + as.numeric(res$beta)
          
          pcc <- cor(y[idx2],ypred,use = "complete")
          pcc
        }
        
        
        res <- c()
        for(i in 1:5){
          idx1 <- which(x != i)
          idx2 <- which(x == i)
          res <- c(res,rrBLUP_func(G_tmp,phe_tmp,idx1,idx2))
        }
        res
      },cl=cl) # lapply
      stopCluster(cl)
      res_one_file <- matrix(apply(results,2,mean),nrow = 1)
      rownames(res_one_file) <- paste0(phe_name)
      print(paste0(phe_name))
      write.table(res_one_file,file = "G3_maize_CV.csv",row.names = T,col.names = F,quote = F,sep = ",",append = T)
}

