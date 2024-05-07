library(childesr)

# Define the output path
path <- "PATH/"

# Define the names of the corpora
corpora <- c('Bates', 'Bernstein', 'Bliss', 'Bloom', 'Bohannon', 'Braunwald', 'Brent', 'Brown', 'Cornell', 'Demetras1', 'Demetras2', 'Feldman', 'Garvey', 'Gathercole', 'Gleason', 'Higginson', 'Kuczaj', 'MacWhinney', 'McCune', 'McMillan', 'Morisset', 'Nelson', 'NewEngland', 'Peters', 'Post', 'Providence', 'Rollins', 'Sachs', 'Sawyer', 'Snow', 'Soderstrom', 'Suppes', 'Tardif', 'VanHouten', 'Warren', 'Weist')

for (corpus in corpora) {
  utts <- get_utterances(corpus=corpus, role_exclude='Target_Child')
  file_path <- paste(path, corpus, ".csv", sep="")
  write.csv(utts, file_path, row.names=FALSE)
}