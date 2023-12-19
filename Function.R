c2f <- function(data){data %>% 
    mutate_if(is.character, funs(as.factor(.)))}

tidy_table <- function(data){
  data %>% 
    mutate(p.value = format_pvalue(p.value)) %>% 
    mutate_if(is.numeric, funs(sprintf("%.2f", .))) %>% 
    mutate(Estimate = glue::glue("{estimate} ({conf.low}–{conf.high})")) %>% 
    dplyr::select(-estimate, -std.error, -statistic,
                  -conf.low, -conf.high)}
