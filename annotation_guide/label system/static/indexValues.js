
const dataFiles = ['medicine','symptom','disease','degree','frequency']
const loadScript = (url) => {
  var script = document.createElement('script')
  script.type = 'text/javascript'
  script.src = url
  document.getElementsByTagName('head')[0].appendChild(script)
}

const indexData = (fileName) => {
  var index = new FlexSearch({
    encode: false,
    suggest: true,
    tokenize: function (str) {
      return str.replace(/[\x00-\x7F]/g, '').split('')
    }
  })
  window[fileName + '_index'] = index
  window[fileName + '_indexCallback'] = (data) => {
    window[fileName + '_indexData'] = data
    for (let i in data) {
      index.add(i, data[i])
    }
  }
  loadScript('./static/' + fileName + '.js')
}
window.indexedValues = {}
for (let i in dataFiles) {
  indexData(dataFiles[i])
}
