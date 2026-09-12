class CapturePCM extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const channels=inputs[0];
    if(channels && channels.length){
      const mono=new Float32Array(channels[0].length);
      for(const channel of channels)for(let i=0;i<mono.length;i++)mono[i]+=channel[i]/channels.length;
      this.port.postMessage(mono);
    }
    for(const channel of outputs[0]||[])channel.fill(0);
    return true;
  }
}
registerProcessor('capture-pcm',CapturePCM);
