# Omics #####
## Normalize #####
map(Dualomics_tbl_list, function(tbl){
  tbl %>%
    remove_all0 %>% 
    remove_iqr0 %>% 
    
    # addmin_log10 %>% 
    group_by(Variable) %>% 
    mutate(Value = bestNormalize::yeojohnson(Value)$x.t) %>% 
    ungroup %>% 
    mutate(Value = 1/Value) %>% 
    
    robustscale %>% 
    outlier2iqr1.5 %>% 
    minmax01 %>% 
    filter(!is.na(Value))
}) -> Dualomics_Scale_tbl_list
# 
# Dualomics_Scale_tbl_list %>%
#   map2(., Dualomics_LRSD_list, function(tbl, sd){
#     tbl %>%
#       filter(Variable %in% sd)
#   }) %>%
#   imap(~ .x %>%
#         mutate(Data = .y)) %>%
#   do.call(bind_rows, .) %>%
#   ggplot(.,
#          aes(x = Value, y = ..scaled.., fill = Variable))+
#   geom_density(alpha = 0.3)+
#   guides(fill = FALSE)+
#   facet_wrap( ~ Data, scale = "free", nrow = 2)

## Logistic regression #####
map(Dualomics_Scale_tbl_list, function(tbl){ # Dualomics_tbl_list
  tbl %>% 
    ungroup %>% 
    inner_join(Asthma_tbw) %>%
    inner_join(Clinicaldata_tbw) %>% 
    group_by(Variable) %>%
    nest() %>%
    
    mutate(Model = map(data, ~ glm(Feature_model,
                                   data = .,
                                   family = "binomial"))) %>%
    mutate(Result = map(Model, ~ broom::tidy(.,
                                             conf.int = TRUE,
                                             exponentiate = TRUE))) %>%
    select(Result) %>%
    unnest() %>%
    filter(term == "Value")}) -> Dualomics_LR_tbw_list

Dualomics_LR_tbw_list %>% 
  map(~ .x %>% 
        filter(p.value < Pvalue_Cutoff1) %>% # FDR < 0.05, n = 0
        check_levels("Variable")) -> Dualomics_LRSD_list

# Product #####
## Product #####
Dualomics_Scale_tbl_list[[1]] %>% 
  filter(Variable %in% Dualomics_LRSD_list[[1]]) %>% 
  spread(Variable, Value) %>% 
  inner_join(Dualomics_Scale_tbl_list[[2]] %>% 
               filter(Variable %in% Dualomics_LRSD_list[[2]]), .) %>% 
  rename(Value_A = Value,
         Variable_A = Variable) %>% 
  gather(Variable_B, Value_B, -Variable_A, -Value_A, -study_id)  %>% 
  mutate(Value = Value_A * Value_B) %>% 
  unite(Variable, Variable_A, Variable_B, sep = "@") %>% 
  select(study_id, Variable, Value) ->  Dualomics_Product_tbl

## Scale #####
Dualomics_Product_tbl %>% 
  remove_all0 %>% 
  remove_iqr0 %>% 
  
  # addmin_log10 %>% 
  group_by(Variable) %>% 
  mutate(Value = bestNormalize::yeojohnson(Value)$x.t) %>% 
  ungroup %>% 
  # mutate(Value = 1/Value) %>% # bad effect for AUC
  
  robustscale %>% 
  outlier2iqr1.5 %>% 
  minmax01 %>% 
  filter(!is.na(Value)) %>% 
  as_tibble -> Dualomics_Product_Scale_tbl

## Logistic regression #####
Dualomics_Product_Scale_tbl %>% # Dualomics_Product_tbl
  inner_join(Asthma_tbw) %>%
  inner_join(Clinicaldata_tbw) %>% 
  group_by(Variable) %>%
  nest() %>%
  mutate(Model = map(data, ~ glm(Feature_model,
                                 data = .,
                                 family = "binomial"))) %>%
  mutate(Result = map(Model, ~ broom::tidy(.,
                                           conf.int = TRUE,
                                           exponentiate = TRUE))) %>%
  select(Result) %>%
  unnest() %>%
  filter(term == "Value") -> Dualomics_Product_LR_tbw

# Dualomics_Product_LR_tbw -> Dualomics_Product_Inv_LR_tbw


Pvalue_Cutoffs <- c(0.05, 0.01,
                    0.005, 0.001,
                    0.0005, 0.0001,
                    0.00005, 0.00001)

map(Pvalue_Cutoffs, function(p){
  Dualomics_Product_LR_tbw %>% 
    ungroup %>% 
    filter(p.value < p) %>% # FDR < 0.05, n = 0
    check_levels("Variable") %>% 
    length}) %>% 
  do.call(c, .) -> Number_Duaomics_Product_LRSD

names(Number_Duaomics_Product_LRSD) <- Pvalue_Cutoffs
Number_Duaomics_Product_LRSD

# Pvalue_Cutoff2 <- 0.0005

Dualomics_Product_LR_tbw %>% 
  ungroup %>% 
  filter(p.value < Pvalue_Cutoff2) %>% # FDR < 0.05, n = 0
  check_levels("Variable") -> Dualomics_Product_LRSD

# FDR_Cutoffs <- c(0.05, 0.01,
#                     0.005, 0.001,
#                     0.0005, 0.0001)

# map(FDR_Cutoffs, function(fdr){
#   Dualomics_Product_LR_tbw %>% 
#     ungroup %>% 
#     mutate(FDR = qvalue_r(p.value)) %>%
#     filter(FDR < fdr) %>% 
#     check_levels("Variable") %>% 
#     length}) %>% 
#   do.call(c, .) -> Number_Duaomics_Product_LRSD_FDR
# 
# names(Number_Duaomics_Product_LRSD_FDR) <- FDR_Cutoffs
# 
# Number_Duaomics_Product_LRSD_FDR
#
# Dualomics_Product_LR_tbw %>%
#   ungroup %>%
#   mutate(FDR = qvalue_r(p.value)) %>%
#   filter(FDR < FDR_Cutoff) %>%
#   check_levels("Variable") -> Dualomics_Product_LRSD_FDR

# Output to CNN #####
list(Omics1 = Dualomics_Scale_tbl_list[[1]] %>% 
       filter(Variable %in% Dualomics_LRSD_list[[1]]),
     
     Omics2 = Dualomics_Scale_tbl_list[[2]] %>% 
       filter(Variable %in% Dualomics_LRSD_list[[2]]),
     
     Dualomics = Dualomics_Product_Scale_tbl %>%
       filter(Variable %in% Dualomics_Product_LRSD)) -> DualomicsProduct_LRSD_tbl_list

DualomicsProduct_LRSD_tbl_list %>% 
  imap(~ .x %>% 
         mutate(Data = .y) %>% 
         unite(Variable, Data, Variable, sep = "__") %>% 
         c2f) %>% 
  do.call(bind_rows, .) %>% 
  spread(Variable, Value) -> DualomicsProduct_tbw
