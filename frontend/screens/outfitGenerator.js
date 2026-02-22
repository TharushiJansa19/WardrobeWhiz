import React from "react";
import { SafeAreaView, StyleSheet, Text, View } from "react-native";
import { Button, IconButton } from "react-native-paper";

const OutfitGeneratorScreen = ({navigation}) => {
  const handleFindMatchingItem = () => {
    navigation.navigate("camera");
  };

  const handleOutfitFinder = () => {
    navigation.navigate("outfit-finder");
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.titleContainer}>
        <Text style={styles.title}>From Items In The Closet</Text>
        <IconButton
          icon="arrow-down"
          color="#F8F0E3"
          size={50}
          style={styles.arrowIcon}
        />
      </View>
      <View style={styles.buttonContainer}>
        <Button
          mode="elevated"
          onPress={handleFindMatchingItem}
          style={styles.button}
          labelStyle={styles.buttonLabel}
          contentStyle={styles.buttonContent}
        >
            Find Matching Item
        </Button>
        <Button
          mode="elevated"
          onPress={handleOutfitFinder}
          style={styles.button}
          labelStyle={styles.buttonLabel}
          contentStyle={styles.buttonContent}
        >
           Outfit Finder By Text
        </Button>
      </View>
    </SafeAreaView>
  );
};

export default OutfitGeneratorScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
    backgroundColor: "#3F301D",
  },
  titleContainer: {
    justifyContent: "center",
    alignItems: "center",
  },
  arrowIcon: {
    marginTop: -20,
    height: 50,
  },
  title: {
    fontSize: 25,
    fontWeight: "bold",
    color: "#F8F0E3",
    marginBottom: 50,
    fontStyle: "italic",
    textShadowColor: "rgba(0, 0, 0, 0.75)", 
    textShadowOffset: { width: -1, height: 1 },
    textShadowRadius: 10,
  },
  
  buttonContainer: {
    width: '100%',
    paddingHorizontal: 20,
    alignItems: 'center'
  },
  button: {
    width: "80%",
    marginVertical: 25,
    paddingVertical: 8,
    backgroundColor:'#4B371C',
  },
  buttonLabel: {
    color: "#FFF",
    fontSize: 18,
    lineHeight: 26,
  },
  buttonContent: {
    height: 50,
  },
});
