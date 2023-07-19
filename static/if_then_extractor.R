#### Regex Extracting ####
RegEx=function(x){
  paste(
    paste("(?<![\\pL\\pM_-])(",
          x,
          ")(?![\\pL\\pM_-])",
          sep = ""),sep="|",collapse = "|"
  )
} 

RegEx_comma=function(x){
  paste(
    paste("(?<=[\\pL\\pM_-])(",
          x,
          ")(?![\\pL\\pM_-])",
          sep = ""),sep="|",collapse = "|"
  )
} 
RegEx_not_by=function(x){
  paste(
    paste("(?<![\\pL\\pM_-])(",
          x,
          ")(?![[:space:]]{1}by[[:space:]]{1})",
          sep = ""),sep="|",collapse = "|"
  )
} 

RegEx_end=function(x){
  paste(
    paste("(?<![\\pL\\pM_-])(",
          x,
          ")$",
          sep = ""),sep="|",collapse = "|"
  )
} 

RegEx_start=function(x){
  paste(
    paste("^(",
          x,
          ")(?![\\pL\\pM_-])",
          sep = ""),sep="|",collapse = "|"
  )
}

#### Sentence tokenizer ####


RegEx_match_naif=function(x){
  paste(x,sep="|",collapse = "|"
  )
}

RegEx_match=function(x){
  paste(
    paste("[^\\pL\\pM_-]",
          x,
          "[^\\pL\\pM_-]",
          sep = ""),
    paste("^",
          x,
          "[^\\pL\\pM_-]",
          sep = ""),
    paste("[^\\pL\\pM_-]",
          x,
          "$",
          sep = ""),
    paste("^",
          x,
          "$",
          sep = ""),
    sep="|",collapse = "|"
  )
}

RegEx_match_start_middle=function(x){
  paste(
    paste("[^\\pL\\pM_-]",
          x,
          "[^\\pL\\pM_-]",
          sep = ""),
    paste("^",
          x,
          "[^\\pL\\pM_-]",
          sep = ""),
    sep="|",collapse = "|"
  )
}


RegEx_match_middle_end=function(x){
  paste(
    paste("[^\\pL\\pM_-]",
          x,
          "[^\\pL\\pM_-]",
          sep = ""),
    paste("[^\\pL\\pM_-]",
          x,
          "$",
          sep = ""),
    sep="|",collapse = "|"
  )
}

RegEx_match_end=function(x){
  paste(
    paste("[^\\pL\\pM_-]",
          x,
          "$",
          sep = ""),
    paste("^",
          x,
          "$",
          sep = "")
    ,sep="|",collapse = "|"
  )
}

RegEx_match_start=function(x){
  paste(
    paste("^",
          x,
          "[^\\pL\\pM_-]",
          sep = ""),
    paste("^",
          x,
          "$",
          sep = ""),
    sep="|",collapse = "|"
  )
}





if_then_extractor_sentence=function(text, 
                                    lang="en",
                                    perl=T,
                                    ignore.case = T)
{
  
  
  if_then_pattern="(?<=^if |[[:space:][:punct:]]if )\\K([[:word:][:space:]'-]){1,}\\K((then |[,] then |[,] ))(?=([[:word:]]{1,}){1,})"
  then_pattern=paste0(RegEx(c("if","then")),"|",RegEx_comma(c("[,] then","[,]")))
  If_=c("if","IF","If")
  Then_=c("then","THEN","Then",",",", then",", THEN",", Then")
  
  
  
  x=list(text)
  # print('TESTO')
  #print(text)
  has_operator = grepl(pattern = if_then_pattern, x, perl=T,ignore.case = ignore.case)
  # print('HERE')
  # print(has_operator)
  operators=list()
  are_if_then=list()
  splits=list()
  IFs=list()
  THENs=list()
  positions=list()
  causal_data=list()
  
  if(has_operator){ 
   
    #print('QUA')
  # if(as.integer((which(has_operator)==0)){print('salta')}
  #### FOR ALL SENTENCES THAT MATCHED THE IF_THEN REGEX  ####
  for (i in which(has_operator)) {
    if(grepl(pattern = if_then_pattern,
             x[[i]],
             perl = perl,
             ignore.case = ignore.case))
    { 
      positions[[i]] = gregexpr(
        pattern = paste(then_pattern),
        x[[i]],
        perl = perl,
        ignore.case = ignore.case
      )
      operators[[i]]	= tolower(regmatches(x[[i]],
                                          positions[[i]])[[1]])
      IFs[[i]]= which(operators[[i]] %in% If_)
      THENs[[i]]= which(operators[[i]] %in% Then_)
      
      ##ONLY ONE IF THEN RELATION
      if(length(IFs[[i]])==length(THENs[[i]])  & length(IFs[[i]])==1  && (IFs[[i]]+1)==THENs[[i]]){
        splits[[i]] = regmatches(x[[i]], positions[[i]],
                                 invert = T)[[1]][-1]
        splits[[i]]=gsub("^([[:punct:][:space:]]){1,}|([[:punct:][:space:]]){1,}$","",splits[[i]])
        causal_data[[i]]=data.frame(IF=splits[[i]][IFs[[i]]], THEN=splits[[i]][THENs[[i]]], SENTENCE=x[[i]], ID=i, stringsAsFactors	= F)
        
      }
      ##MULTIPLE IF THEN RELATIONS (CORRECTLY FORMED)
      if(length(IFs[[i]])==length(THENs[[i]]) & identical(IFs[[i]]+1,THENs[[i]])){
        splits[[i]] = regmatches(x[[i]], positions[[i]],
                                 invert = T)[[1]][-1]
        splits[[i]]=gsub("^([[:punct:][:space:]]){1,}|([[:punct:][:space:]]){1,}$","",splits[[i]])
        #splits[[i]]=gsub("^(and )|( and)$","",splits[[i]])
        causal_data[[i]]=data.frame(IF=splits[[i]][IFs[[i]]], THEN=splits[[i]][THENs[[i]]], SENTENCE=x[[i]], ID=i, stringsAsFactors	= F)
      }
      ##MULTIPLE IF THEN RELATIONS (ADVANCED PAIRING)
      if(length(IFs[[i]])!=length(THENs[[i]]) | !identical(IFs[[i]]+1,THENs[[i]])){
        IFs[[i]]=intersect(IFs[[i]]+1,THENs[[i]])-1
        THENs[[i]]=intersect(IFs[[i]]+1,THENs[[i]])
        splits[[i]] = regmatches(x[[i]], positions[[i]],
                                 invert = T)[[1]][-1]
        splits[[i]]=gsub("^([[:punct:][:space:]]){1,}|([[:punct:][:space:]]){1,}$","",splits[[i]])
        #splits[[i]]=gsub("^(and )|( and)$","",splits[[i]])
        causal_data[[i]]=data.frame(IF=splits[[i]][IFs[[i]]], THEN=splits[[i]][THENs[[i]]], SENTENCE=x[[i]], stringsAsFactors	= F)
      }
    }
  }
  if(exists("causal_data")){
    #print(causal_data)
    ifres = causal_data[[1]]$IF
    thenres = causal_data[[1]]$THEN
    #print(ifres)
    #print(thenres)
    # return(list(causal_data,ifres,thenres))
  }
  }else{
    causal_data = '000000'
    ifres = '000000'
    thenres = '000000'
    # print('noooooooooo')
    
    }
 
  # {causal_data = 'VUOTO'
  #  ifres = 'VUOTO'
  # thenres = 'VUOTO'}
   
    return(list(causal_data,ifres,thenres))

     
   
    
  #  }
    
  
}

