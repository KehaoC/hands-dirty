// 配置 Babel 转译器，决定你的 JS/TS 代码怎么被“翻译”成能在手机上跑的代码。

module.exports = {
    presets: ['babel-preset-expo'],
    plugins: [
      // 其他插件可以放这里
      'react-native-reanimated/plugin', // 这个一定要放最后
    ],
  };