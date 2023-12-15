# Clinical #####
## Scale #####
Clinical_tbl %>% 
  filter(Variable %in% Variable_Con)  %>% 
  
  # addmin_log10 %>% 
  ungroup %>% 
  group_by(Variable) %>% 
  mutate(Value = bestNormalize::yeojohnson(Value)$x.t) %>% 
  ungroup %>% 
  
  robustscale %>% 
  outlier2iqr1.5 %>% 
  minmax01 %>% 
  bind_rows(Clinical_tbl %>% 
              filter(!Variable %in% Variable_Con)) -> Clinical_Scale_tbl

Clinical_tbl %>% 
  filter(Variable %in% Variable_Con)  %>% 
  
  # addmin_log10 %>% 
  ungroup %>% 
  group_by(Variable) %>% 
  mutate(Value = bestNormalize::yeojohnson(Value)$x.t) %>% 
  ungroup %>% 
  mutate(Value = 1/Value) %>% 
  
  robustscale %>% 
  outlier2iqr1.5  %>% 
  minmax11 %>% 
  bind_rows(Clinical_tbl %>% 
              filter(!Variable %in% Variable_Con) %>% 
              mutate(Value = if_else(Value == 0, -1, Value))) -> Clinical_ScaleOne_tbl


# Clinical-Singleomics product #####
## Product
map2(Dualomics_Scale_tbl_list, Dualomics_LRSD_list,
     function(tbl, var){
       tbl %>% 
         filter(Variable %in% var) %>% 
         spread(Variable, Value) %>% 
         inner_join(Clinical_ScaleOne_tbl, .) %>% # Clinical_Scale_tbl
         rename(Value_A = Value,
                Variable_A = Variable) %>% 
         gather(Variable_B, Value_B, -Variable_A, -Value_A, -study_id)  %>% 
         mutate(Value = Value_A * Value_B) %>% 
         unite(Variable, Variable_A, Variable_B, sep = "--") %>% 
         select(study_id, Variable, Value)}) -> ClinicalOmics_Product_tbl_list

## Scale #####
map(ClinicalOmics_Product_tbl_list,
    function(tbl){
      tbl %>%
        
        # addmin_log10 %>% 
        ungroup %>% 
        group_by(Variable) %>% 
        mutate(Value = bestNormalize::yeojohnson(Value)$x.t) %>% 
        ungroup %>% 
        
        robustscale %>% 
        outlier2iqr1.5 %>% 
        minmax01 %>% 
        filter(!is.na(Value))}) -> ClinicalOmics_Product_Scale_tbl_list


## Logistic regression #####
ClinicalOmics_Product_Scale_tbl_list %>% # ClinicalOmics_Product_tbl_list
  map(~ .x %>% 
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
        select(Variable, Result) %>%
        select(Result) %>%
        unnest() %>%
        filter(term == "Value")) -> ClinicalOmics_Product_LR_tbw_list


ClinicalOmics_Product_LR_tbw_list %>% 
  map(~ .x %>% 
        ungroup %>% 
        # mutate(FDR = qvalue_r(p.value)) %>% 
        filter(p.value < Pvalue_Cutoff2) %>% # FDR < 0.05, n = 0
        # filter(FDR < 0.05) %>% 
        check_levels("Variable")) -> ClinicalOmics_Product_LRSD_list

# Clinical-Dualomics product #####
## Product #####
Dualomics_Product_Scale_tbl %>%
  filter(Variable %in% Dualomics_Product_LRSD) %>% # Place after TAG lipids
  spread(Variable, Value) %>%
  inner_join(Clinical_ScaleOne_tbl, .) %>% # Clinical_Scale_tbl
  rename(Value_A = Value,
         Variable_A = Variable) %>%
  gather(Variable_B, Value_B, -Variable_A, -Value_A, -study_id)  %>%
  mutate(Value = Value_A * Value_B) %>%
  unite(Variable, Variable_A, Variable_B, sep = "--") %>%
  select(study_id, Variable, Value) %>%
  c2f ->  ClinicalDualomics_Product_tbl

## Scale #####
ClinicalDualomics_Product_tbl %>% 
  filter(is.na(Value))

ClinicalDualomics_Product_tbl %>% 
  remove_all0 %>% 
  remove_iqr0 %>% 
  
  # addmin_log10 %>% 
  ungroup %>% 
  group_by(Variable) %>% 
  mutate(Value = bestNormalize::yeojohnson(Value)$x.t) %>% 
  ungroup %>% 
  
  robustscale %>% 
  outlier2iqr1.5 %>% 
  minmax01 -> ClinicalDualomics_Product_Scale_tbl


## Logistic regression #####
ClinicalDualomics_Product_Scale_tbl %>% # ClinicalDualomics_Product_tbl
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
  select(Variable, Result) %>%
  select(Result) %>%
  unnest() %>%
  filter(term == "Value") -> ClinicalDualomics_Product_LR_tbw

ClinicalDualomics_Product_LR_tbw %>%
  # filter(p.value < Pvalue_Cutoff2) %>% # FDR < 0.05, n = 0
  # filter(p.value < 0.001) %>% # FDR < 0.05, n = 0
  filter(p.value < Pvalue_Cutoff3) %>% # FDR < 0.05, n = 0
  ungroup %>% 
  # mutate(FDR = qvalue_r(p.value)) %>% 
  # filter(FDR < 0.01) %>% 
  check_levels("Variable") -> ClinicalDualomics_Product_LRSD

# Output to CNN #####
list(Omics1 = Dualomics_Scale_tbl_list[[1]] %>% 
       filter(Variable %in% Dualomics_LRSD_list[[1]]),
     
     Omics2 = Dualomics_Scale_tbl_list[[2]] %>% 
       filter(Variable %in% Dualomics_LRSD_list[[2]]),
     
     Clinical = Clinical_Scale_tbl,
     
     Dualomics = Dualomics_Product_Scale_tbl %>%
                  filter(Variable %in% Dualomics_Product_LRSD),
     
     ClinicalOmics1 = ClinicalOmics_Product_Scale_tbl_list[[1]] %>%
                    filter(Variable %in% ClinicalOmics_Product_LRSD_list[[1]]),
     
     ClinicalOmics2 = ClinicalOmics_Product_Scale_tbl_list[[2]] %>%
                    filter(Variable %in% ClinicalOmics_Product_LRSD_list[[2]]),
     
     ClinicalDualomics = ClinicalDualomics_Product_Scale_tbl %>%
                    filter(Variable %in% ClinicalDualomics_Product_LRSD)) -> ClinicalDualomicsProduct_LRSD_tbl_list

ClinicalDualomicsProduct_LRSD_tbl_list %>% 
  imap(~ .x %>% 
         mutate(Data = .y) %>% 
         unite(Variable, Data, Variable, sep = "__") %>% 
         c2f) %>% 
  do.call(bind_rows, .) %>% 
  spread(Variable, Value) -> ClinicalDualomicsProduct_tbw
