import React from "react";
import { View, TouchableOpacity, Text, Image, StyleSheet } from "react-native";
import { ActivityIndicator, IconButton } from "react-native-paper";

const CARD_WIDTH = "47%";
const CARD_HEIGHT = 200;

const renderContent = (items, handleAddItems, isLoading) => {
  if (isLoading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size={60} animating={true} color="#765952" />
      </View>
    );
  }
  if (items.length === 0) {
    return (
      <TouchableOpacity
        style={styles.watermarkContainer}
        onPress={handleAddItems}
      >
        <IconButton
          icon="plus"
          size={48}
          style={styles.watermarkIcon}
          color="#cccccc"
        />
        <Text style={styles.watermarkText}>Add your clothes</Text>
      </TouchableOpacity>
    );
  } else {
    return items.map((item, index) => (
      <View key={index} style={styles.card}>
        <Image
          source={{ uri: item }}
          style={{ width: "100%", height: "100%" }}
          resizeMode="contain"
        />
      </View>
    ));
  }
};

const styles = StyleSheet.create({
  card: {
    marginBottom: 10,
    elevation: 4,
    backgroundColor: "#F8F0E3",
    width: CARD_WIDTH,
    height: CARD_HEIGHT,
  },
  watermarkContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingTop: 50,
  },
  watermarkIcon: {
    backgroundColor: "#e0e0e0",
    borderRadius: 24,
  },
  watermarkText: {
    color: "#cccccc",
    fontSize: 16,
    marginTop: 10,
  },
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingTop: 60,
  },
});

export default renderContent;
