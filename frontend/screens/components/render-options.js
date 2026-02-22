import React from "react";
import { TouchableOpacity, Text, View, StyleSheet } from "react-native";

const renderOptions = (
  selectedSegmentIndex,
  categories,
  colors,
  handleCategoryPress,
  handleColorPress
) => {
  if (selectedSegmentIndex === 0) {
    return categories.map((category, index) => (
      <TouchableOpacity
        key={index}
        style={styles.categoryIcon}
        onPress={() => handleCategoryPress(category)}
      >
        <Text style={styles.categoryText}>{category}</Text>
      </TouchableOpacity>
    ));
  } else {
    return colors.map((color, index) => (
      <TouchableOpacity key={index} style={styles.categoryIcon} onPress={() => handleColorPress(color)}>
        <View
          style={[styles.colorCircle, { backgroundColor: color.toLowerCase() }]}
        />
        <Text style={styles.categoryText}>{color}</Text>
      </TouchableOpacity>
    ));
  }
};

const styles = StyleSheet.create({
  categoryIcon: {
    alignItems: "center",
    justifyContent: "center",
    marginHorizontal: 10,
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "#F8F0E3",
    elevation: 4,
  },
  colorCircle: {
    width: 10,
    height: 10,
    marginBottom: 5,
  },
  categoryText: {
    textAlign: "center",
    color: "#4A3728",
    fontWeight: "bold",
    textTransform: "capitalize"
  },
});

export default renderOptions;
