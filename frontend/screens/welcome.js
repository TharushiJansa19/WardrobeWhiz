import { ImageBackground, SafeAreaView, StyleSheet, View } from "react-native";
import React from "react";
import { FontAwesome } from "@expo/vector-icons";
import { AntDesign } from "@expo/vector-icons";
import { StatusBar } from "expo-status-bar";
import { Button, Text } from "react-native-paper";

const WelcomeScreen = ({ navigation }) => {
  const handleSignIn = () => {
    navigation.navigate("signIn");
  };

  const handleRegister = () => {
    navigation.navigate("register");
  };

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <ImageBackground
        source={require("../assets/welcome_bg.webp")}
        style={styles.container}
        resizeMode="cover"
      >
        <View style={{ marginTop: 60 }}>
          <Text style={styles.welcomeText} variant="headlineLarge">
            Welcome
          </Text>
          <Text style={styles.text} variant="headlineLarge">
            WARDROBE WHIZ
          </Text>
        </View>

        <View style={styles.bottomContainer}>
          <View style={styles.subtitleContainer}>
            <Text style={styles.description} variant="bodyMedium">
              AI Powered Wardrobe Management and Shopping Assistant
            </Text>
          </View>
          <Button
            mode="elevated"
            buttonColor="#765952"
            style={{ borderRadius: 0, marginBottom: 15 }}
            uppercase={true}
            labelStyle={{ fontSize: 15 }}
            textColor="#fff"
            onPress={handleRegister}
          >
            Register
          </Button>

          <Button
            mode="elevated"
            buttonColor="#765952"
            style={{ borderRadius: 0 }}
            uppercase={true}
            labelStyle={{ fontSize: 15 }}
            textColor="#fff"
            onPress={handleSignIn}
          >
            Sign In
          </Button>
        </View>
      </ImageBackground>
    </SafeAreaView>
  );
};

export default WelcomeScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "space-between",
  },

  text: {
    color: "#fff",
    textAlign: "center",
    fontWeight: "bold",
    fontSize: 35,
    lineHeight: 60,
    letterSpacing: 1,
    textShadowColor: "#9A7B4F",
    textShadowOffset: { width: -2, height: 2 },
    textShadowRadius: 15,
    shadowOpacity: 4,
    elevation: 8,
  },

  welcomeText: {
    fontSize: 25,
    color: "white",
    textAlign: "center",
    letterSpacing: 1,
    fontStyle: "italic",
  },
  description: {
    color: "white",
    textAlign: "center",
    fontStyle: "italic",
  },
  subtitleContainer: {
    marginBottom: 20,
    alignSelf: "center",
    borderTopWidth: 0.5,
    borderColor: "white",
    borderBottomWidth: 0.5,
    height: 60,
    width: 250,
    backgroundColor: "transparent",
    justifyContent: "center",
  },
  bottomContainer: {
    marginBottom: 30,
  },
});
