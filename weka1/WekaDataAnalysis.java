import weka.core.Instances;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.converters.ConverterUtils.DataSource;


public class WekaDataAnalysis {
    
    public static void main(String[] args) {
        // Argumentuak egiaztatu
        if (args.length != 1) {
            System.err.println("Errorea: ARFF fitxategiaren path-a eman behar da argumentu gisa.");
            System.err.println("Erabilera: java WekaDataAnalysis <fitxategia.arff>");
            System.exit(1);
        }
        
        String fitxategiPath = args[0];
        
        try {
            // Datu-sorta kargatu
            System.out.println("========================================");
            System.out.println("WEKA DATU-ANALISI GAINAZALEKOA");
            System.out.println("========================================\n");
            
            DataSource source = new DataSource(fitxategiPath);
            Instances data = source.getDataSet();
            
            // Klasea azken atributuan ezarri
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            
            // 1. Fitxategiaren path-a
            System.out.println("1. FITXATEGIAREN PATH-A:");
            System.out.println("   " + fitxategiPath + "\n");
            
            // 2. Instantzia kopurua
            int instantziaKop = data.numInstances();
            System.out.println("2. INSTANTZIA KOPURUA:");
            System.out.println("   " + instantziaKop + " instantzia\n");
            
            // 3. Atributu kopurua
            int atributuKop = data.numAttributes();
            System.out.println("3. ATRIBUTU KOPURUA:");
            System.out.println("   " + atributuKop + " atributu\n");
            
            // 4. Lehenengo atributuaren balio ezberdinak
            analizatuLehenengoAtributua(data);
            
            // 5. Azken atributuaren (klasearen) balioak eta maiztasunak
            analizatuKlaseAtributua(data);
            
            // 6. Azken aurreko atributuaren missing values
            analizatuAzkenAurrekoAtributua(data);
            
            System.out.println("========================================");
            System.out.println("ANALISIA BUKATUTA");
            System.out.println("========================================");
            
        } catch (Exception e) {
            System.err.println("Errorea datu-sorta kargatzerakoan: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    /**
     * Lehenengo atributuaren analisia egiten du
     */
    private static void analizatuLehenengoAtributua(Instances data) {
        Attribute lehenengoAtrib = data.attribute(0);
        
        System.out.println("4. LEHENENGO ATRIBUTUAREN INFORMAZIOA:");
        System.out.println("   Izena: " + lehenengoAtrib.name());
        System.out.println("   Mota: " + Attribute.typeToString(lehenengoAtrib.type()));
        
        if (lehenengoAtrib.isNominal()) {
            int balioKop = lehenengoAtrib.numValues();
            System.out.println("   Balio ezberdin kopurua: " + balioKop);
            System.out.print("   Balioak: ");
            for (int i = 0; i < balioKop; i++) {
                System.out.print(lehenengoAtrib.value(i));
                if (i < balioKop - 1) {
                    System.out.print(", ");
                }
            }
            System.out.println("\n");
        } else if (lehenengoAtrib.isNumeric()) {
            AttributeStats stats = data.attributeStats(0);
            System.out.println("   Balio ezberdin kopurua: " + stats.distinctCount);
            System.out.println("   (Atributu numerikoa da)\n");
        } else {
            System.out.println("   (Beste mota bateko atributua)\n");
        }
    }
    
    /**
     * Klase atributuaren (azken atributua) analisia egiten du
     */
    private static void analizatuKlaseAtributua(Instances data) {
        int klaseanIndex = data.classIndex();
        Attribute klaseAtrib = data.classAttribute();
        AttributeStats klaseStats = data.attributeStats(klaseanIndex);
        
        System.out.println("5. KLASE ATRIBUTUAREN INFORMAZIOA (azken atributua):");
        System.out.println("   Izena: " + klaseAtrib.name());
        System.out.println("   Klase kopurua: " + klaseAtrib.numValues());
        System.out.println("\n   KLASE BAKOITZAREN MAIZTASUNAK:");
        
        int minMaiztasuna = Integer.MAX_VALUE;
        String klaseMinoritarioa = "";
        
        // Klase bakoitzaren maiztasuna inprimatu
        for (int i = 0; i < klaseAtrib.numValues(); i++) {
            String klaseIzena = klaseAtrib.value(i);
            int maiztasuna = klaseStats.nominalCounts[i];
            double ehunekoa = (maiztasuna * 100.0) / data.numInstances();
            
            System.out.printf("   - %s: %d instantzia (%.2f%%)\n", 
                            klaseIzena, maiztasuna, ehunekoa);
            
            // Klase minoritarioa aurkitu
            if (maiztasuna < minMaiztasuna) {
                minMaiztasuna = maiztasuna;
                klaseMinoritarioa = klaseIzena;
            }
        }
        
        System.out.println("\n   KLASE MINORITARIOA:");
        System.out.printf("   '%s' (maiztasuna: %d)\n\n", klaseMinoritarioa, minMaiztasuna);
    }
    
    /**
     * Azken aurreko atributuaren missing values analisia egiten du
     */
    private static void analizatuAzkenAurrekoAtributua(Instances data) {
        int azkenAurrekoIndex = data.numAttributes() - 2;
        
        if (azkenAurrekoIndex < 0) {
            System.out.println("6. AZKEN AURREKO ATRIBUTUA:");
            System.out.println("   Ez dago azken aurreko atributurik (datu-sortak atributu bat bakarrik du).\n");
            return;
        }
        
        Attribute azkenAurrekoAtrib = data.attribute(azkenAurrekoIndex);
        AttributeStats stats = data.attributeStats(azkenAurrekoIndex);
        
        System.out.println("6. AZKEN AURREKO ATRIBUTUAREN INFORMAZIOA:");
        System.out.println("   Izena: " + azkenAurrekoAtrib.name());
        System.out.println("   Mota: " + Attribute.typeToString(azkenAurrekoAtrib.type()));
        System.out.println("   Missing values kopurua: " + stats.missingCount);
        
        if (stats.missingCount > 0) {
            double ehunekoa = (stats.missingCount * 100.0) / data.numInstances();
            System.out.printf("   Missing values ehunekoa: %.2f%%\n", ehunekoa);
        }
        System.out.println();
    }
}