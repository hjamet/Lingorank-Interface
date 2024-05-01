import logging
import src.Config as Config
from colorama import Fore, Style, init

# Initialiser colorama
init(autoreset=True)

# Créer un formateur personnalisé
formatter = logging.Formatter(
    f"{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.YELLOW}%(filename)s{Style.RESET_ALL}:{Fore.CYAN}%(lineno)d{Style.RESET_ALL} - {Fore.RED}%(levelname)s{Style.RESET_ALL} - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Créer un gestionnaire de sortie console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Obtenir le logger racine
logger = logging.getLogger()
logger.setLevel(Config.log_level)

# Ajouter le gestionnaire au logger
logger.addHandler(console_handler)
