import * as React from "react";
import {
  View,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  TouchableOpacity,
} from "react-native";
import { TextInput, Button, Text } from "react-native-paper";
import { FontAwesome5 } from "@expo/vector-icons";
import { MaterialIcons } from "@expo/vector-icons";
import axios from "axios";
import { AUTH_BASE_URL } from "../lib/url";

const SignInScreen = ({ navigation }) => {
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  

  const handleLogin = async () => {
    try {
      const response = await axios.post(
        `${AUTH_BASE_URL}/login`,
        { email, password },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      console.log("Login successful.", response.data);

      const userName = response.data.user;
      navigation.navigate("home", { user: userName });

    } catch (error) {
      let message = "Login failed: ";
      if (error.response) {
          message += error.response.data.message.message || "An error occurred during login.";
          console.error("Login failed:", error.response.data);
      } else if (error.request) {
          message += "No response received from the server.";
          console.error("Login failed:", error.request);
      } else {
          message += error.message;
          console.error("Error", error.message);
      }
      alert(message);
    }
    // navigation.navigate("home", { user: userName = "Guest" });
  };
  

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <View style={styles.innerContainer}>
        <Text style={styles.title}>Sign In</Text>
        <TextInput
          label="Email"
          value={email}
          onChangeText={(text) => setEmail(text)}
          mode="flat"
          underlineColor="#fff"
          style={styles.input}
          theme={{
            colors: { primary: "#765952", underlineColor: "transparent" },
          }}
        />
        <TextInput
          label="Password"
          value={password}
          onChangeText={(text) => setPassword(text)}
          mode="flat"
          underlineColor="#fff"
          secureTextEntry
          style={styles.input}
          theme={{
            colors: { primary: "#765952", underlineColor: "transparent" },
          }}
          right={<TextInput.Icon name="eye-off-outline" />}
        />
        <TouchableOpacity style={styles.forgotPassword}>
          <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
        </TouchableOpacity>
        <Button
          mode="elevated"
          onPress={handleLogin}
          style={styles.button}
          labelStyle={styles.buttonText}
        >
          LOGIN
        </Button>
        <TouchableOpacity
          style={styles.signUp}
          onPress={() => navigation.navigate("register")}
        >
          <Text style={styles.signUpText}>
            Don't have an Account?{" "}
            <Text style={styles.registerText}>Register</Text>
          </Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fbf8f6",
  },
  innerContainer: {
    flex: 1,
    paddingHorizontal: "5%",
    justifyContent: "center",
  },
  title: {
    fontSize: 30,
    fontWeight: "bold",
    marginBottom: 40,
    alignSelf: "center",
    letterSpacing: 1,
    color: "#000",
  },
  input: {
    marginBottom: 15,
    backgroundColor: "#fff",
    elevation: 4,
  },
  button: {
    marginTop: 30,
    borderRadius: 25,
    backgroundColor: "#765952",
    paddingVertical: 4,
  },
  buttonText: {
    color: "#FFF",
    fontWeight: "bold",
  },
  forgotPassword: {
    alignSelf: "flex-end",
    marginVertical: 10,
  },
  forgotPasswordText: {
    color: "#765952",
    fontWeight: "bold",
  },
  signUp: {
    position: "absolute",
    bottom: 25,
    alignSelf: "center",
    width: "100%",
    alignItems: "center",
  },
  signUpText: {
    color: "#000",
  },
  registerText: {
    fontWeight: "bold",
    fontSize: 16,
    color: "#765952",
  },
});

export default SignInScreen;
