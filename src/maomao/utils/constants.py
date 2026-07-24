PATH_INPUT = "../../raw_data"
PATH_EXPORT = "../../processed_data"

MIN_LENGTH_SEQUENCE = 5
MAX_LENGTH_SEQUENCE = 70

CANONICAL_RESIDUES = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                      "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

COLUMNS_TO_WORK = [
    "name dataset", 
    "name source", 
    "type source", 
    "static-dynamic",	
    "license",	
    "reports constant updates", 
    "year of publication",	
    "last update date",	
    "download date",	
    "file format",	
    "protein format",	
    "category dataset",	
    "task",	
    "obtaining negative dataset",	
    "obtaining positive dataset",	
    "repository or server",	
    "publication"
]

BASE_URL_PDB = "https://files.rcsb.org/download/"
BASE_URL_ALPHAFOLD = "https://alphafold.ebi.ac.uk/files"
BASE_URL_UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/"
TIMEOUT: int = 120
