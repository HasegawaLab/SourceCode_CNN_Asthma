source("/Users/shibataryohei/Git/Database/Function/Function.R")
source("/Users/shibataryohei/Git/RS_MARC35/format_data_forestplot.R")

# Colors for color blindness
Colors_Endotype <- c("#E69F00", # orange
                     "#56B4E9", # lightblue
                     "#009E73", # green
                     "#F0E442", # yellow
                     "#0072B2", # blue
                     "#D55E00", # red
                     "#CC79A7") # pink
                     

Colors_Mycotype <- c("#E69F00", # orange
                              "#56B4E9", # lightblue
                              "#009E73", # green
                              "#F0E442", # yellow
                              "#0072B2", # blue
                              "#D55E00", # red
                              "#CC79A7") # pink

Colors_Phenotype <- c("#E69F00", # orange
                     "#56B4E9", # lightblue
                     "#009E73", # green
                     "#F0E442", # yellow
                     "#0072B2", # blue
                     "#D55E00", # red
                     "#CC79A7") # pink


filter_mb <- function(tbl,
                      rate){
  tbl %>% 
    check_levels("Variable") %>% 
    .[1] -> factor
  
  tbl %>% 
    filter(Variable == factor) %>% 
    nrow -> n
  
  tbl %>% 
    filter(Value > 0) %>% 
    group_by(Variable) %>% 
    dplyr::summarise(Count = n()) %>% 
    filter(Count > n * (rate)) %>% 
    check_levels("Variable") -> include_variable
  
  tbl %>% 
    filter(Variable %in% include_variable)}