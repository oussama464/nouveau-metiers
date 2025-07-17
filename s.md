Nous rencontrons une erreur lors de l'exécution de SonarScanner sur Windows, liée à une incompatibilité de version Java.
Le scanner échoue avec l’erreur UnsupportedClassVersionError, indiquant que certaines classes ont été compilées avec Java 17,
tandis que la version actuellement utilisée est Java 11 (fournie dans le dossier jre du scanner).

Comme Java 11 ne peut pas exécuter du bytecode compilé avec Java 17, l’analyse échoue immédiatement.

Pour résoudre ce problème, nous aurions besoin que l’équipe support mette à disposition un JDK 17 pour Windows au format ZIP,
afin que nous puissions remplacer proprement le répertoire jre du scanner par une version compatible.

Une autre option serait de réinstaller SonarScanner sur BZZC avec une configuration adaptée.
