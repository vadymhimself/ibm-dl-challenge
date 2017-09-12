
# load the train meta data
train_meta <- read.csv("data/train.csv", header = T)
# see what labels do we have in the data set
labels <- levels(train_meta$label)

set.seed(1337)

# define a function for predicting random labels 
predict_random <- function(image_ids) {
  return(sample(labels, size = length(image_ids), replace = T))
}

# load test data
test_meta <- read.csv("data/test.csv", header = T)

# predict random labels for the test set
test_meta$label <- predict_random(test_meta$image_id)

# export prediction to the csv file
write.csv(test_meta, file = "submission.csv", row.names = F)
