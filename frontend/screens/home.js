import React from "react";
import { View, StyleSheet, SafeAreaView, TouchableOpacity } from "react-native";
import { Card, Text, Avatar } from "react-native-paper";

const HomeScreen = ({ navigation, route }) => {
  const { user } = route.params;

  const firstName = user ? user.split(" ")[0] : "Guest";

  const userInitial = firstName.charAt(0).toUpperCase();

  const handleMyCloset = () => {
    navigation.navigate("closet");
  };

  const handlePurchaseItem = () => {
    navigation.navigate("camera");
  };

  const handleOutfitRecommendation = () => {
    navigation.navigate("outfit-generator");
  };

  const buttons = [
    {
      title: "My Closet",
      onPress: handleMyCloset,
      backgroundColor: "#C0804019",
    },
    {
      title: "Purchase Item",
      onPress: handlePurchaseItem,
      backgroundColor: "#B4653519",
    },
    {
      title: "Outfit Recommendation",
      onPress: handleOutfitRecommendation,
      backgroundColor: "#D8A06006",
    },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <View>
          <Text variant="displaySmall" style={styles.greeting}>
            Hello
          </Text>
          <Text variant="headlineMedium" style={styles.userName}>
            {firstName}
          </Text>
        </View>
        <Avatar.Text
          size={40}
          label={userInitial}
          style={styles.avatar}
          labelStyle={styles.avatarLabel}
          color={"#fff"}
          backgroundColor={"#4B371C"}
        />
      </View>
      <View style={styles.buttonContainer}>
        {buttons.map((button, index) => (
          <TouchableOpacity
            key={index}
            onPress={button.onPress}
            style={styles.touchable}
          >
            <Card
              style={[styles.card, { backgroundColor: button.backgroundColor }]}
            >
              <Text style={styles.buttonText}>{button.title}</Text>
            </Card>
          </TouchableOpacity>
        ))}
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FFF",
  },
  header: {
    alignItems: "center",
    marginVertical: 80,
    marginHorizontal: 25,
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
  },
  avatar: {
    width: 70,
    height: 70,
    borderRadius: 35,
  },
  avatarLabel: {
    fontWeight: "bold",
    fontSize: 28,
  },
  greeting: {
    fontWeight: "bold",
    letterSpacing: 0.8,
  },
  userName: {
    fontWeight: "600",
    fontStyle: "italic",
    letterSpacing: 1.5,
    marginTop: 8,
  },
  buttonContainer: {
    alignItems: "center",
  },
  card: {
    flex: 1,
    borderRadius: 6,
    borderColor: "#765952",
    justifyContent: "center",
    alignItems: "center",
    height: 125,
    // marginVertical: 20,
  },
  touchable: {
    borderRadius: 8,
    width: "80%",
    height: 125,
    marginVertical: 20,
  },
  buttonText: {
    color: "#000",
    fontSize: 18,
    fontWeight: "bold",
    textTransform: "uppercase",
  },
});

export default HomeScreen;
